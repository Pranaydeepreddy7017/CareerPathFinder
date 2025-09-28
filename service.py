# service.py
from typing import Dict, List
import pandas as pd
import torch
from core import load_artifacts, get_driver, settings

# service.py (add this)
from typing import List, Dict
from neo4j import GraphDatabase
from core import settings, get_driver

driver = get_driver()

model, scaler, feature_columns = load_artifacts()
driver = get_driver()

def _difficulty_label(num):
    if num is None: return "Unknown"
    if num <= 2: return "Easy"
    if num == 3: return "Medium"
    return "Hard"

def predict_student_course(student_id: str, course_id: str) -> Dict:
    with driver.session(database=settings.DB_NAME) as s:
        student_data = s.run("""
        MATCH (s:Student {id:$sid})-[:COMPLETED]->(c:Course)
        RETURN avg(c.difficulty) AS avg_difficulty,
               avg(CASE WHEN c.grade IN ['A','A-','B+','B'] THEN 1.0 ELSE 0 END) AS pass_rate,
               avg(c.timeSpent) AS avg_time,
               s.learningStyle AS learningStyle,
               s.preferredCourseLoad AS courseLoad,
               s.preferredPace AS pace
        """, sid=student_id).single()

        course_data = s.run(
            "MATCH (c:Course {id:$cid}) RETURN c.level AS difficulty",
            cid=course_id
        ).single()

    if not student_data or not course_data:
        return {"error": f"Missing data for {student_id} or {course_id}"}

    row = {
        "difficulty": course_data["difficulty"] or 3,
        "timeSpent": student_data["avg_time"] or 6,
        "learningStyle": student_data["learningStyle"],
        "courseLoad": student_data["courseLoad"],
        "pace": student_data["pace"],
        "past_gpa": (student_data["pass_rate"] or 0.6) * 4.0,
        "recent_gpa": (student_data["pass_rate"] or 0.6) * 4.0,
        "avg_difficulty": student_data["avg_difficulty"] or 3,
        "credits_taken": 10,
    }

    df = pd.get_dummies(pd.DataFrame([row])).reindex(columns=feature_columns, fill_value=0)
    prob = model(torch.tensor(scaler.transform(df), dtype=torch.float32)).item()

    return {
        "student_id": student_id,
        "course_id": course_id,
        "probability": round(prob, 2),
        "result": "Likely to Succeed" if prob >= 0.5 else "At Risk of Failing",
    }

def recommend_courses(student_id: str, term: str = "Spring2024",
                      allow_cross_dept: bool = False, max_courses: int = 10) -> Dict:
    from typing import Dict as _Dict  # avoid circular type references in editors

    with driver.session(database=settings.DB_NAME) as s:
        student = s.run("""
        MATCH (s:Student {id:$sid})-[:PURSUING]->(d:Degree)
        RETURN s.id AS id, s.learningStyle AS learningStyle,
               s.preferredCourseLoad AS courseLoad, s.preferredPace AS pace,
               s.enrollmentDate AS enrollmentDate, s.expectedGraduation AS gradDate,
               d.name AS degreeName, d.id AS degreeId, d.totalCreditsRequired AS totalCredits
        """, sid=student_id).single()
        if not student:
            return {"error": f"No student {student_id}"}

        hist = s.run("""
        MATCH (s:Student {id:$sid})-[c:COMPLETED]->(course:Course)
        RETURN avg(c.difficulty) AS avg_difficulty,
               avg(CASE WHEN c.grade IN ['A','A-','B+','B'] THEN 1.0 ELSE 0 END) AS pass_rate,
               avg(c.timeSpent) AS avg_time,
               sum(course.credits) AS credits_completed,
               count(c) AS courses_completed
        """, sid=student_id).single()

        if allow_cross_dept:
            cq = """
            MATCH (c:Course)-[:OFFERED_IN]->(t:Term {id:$term})
            WHERE NOT EXISTS { MATCH (:Student {id:$sid})-[:COMPLETED]->(c) }
            RETURN c.id AS courseId, c.name AS name, c.credits AS credits,
                   c.avgDifficulty AS difficulty, c.department AS dept
            """
        else:
            cq = """
            MATCH (s:Student {id:$sid})-[:PURSUING]->(d:Degree)<-[:PART_OF]-(r:RequirementGroup)
                  <-[:FULFILLS]-(c:Course)-[:OFFERED_IN]->(t:Term {id:$term})
            WHERE NOT EXISTS { MATCH (:Student {id:$sid})-[:COMPLETED]->(c) }
            RETURN DISTINCT c.id AS courseId, c.name AS name, c.credits AS credits,
                            c.avgDifficulty AS difficulty, c.department AS dept
            """
        courses = [rec.data() for rec in s.run(cq, sid=student_id, term=term)]

    recs: List[_Dict] = []
    for c in courses:
        row = {
            "difficulty": c["difficulty"] or 3,
            "timeSpent": hist["avg_time"] or 6,
            "learningStyle": student["learningStyle"],
            "courseLoad": student["courseLoad"],
            "pace": student["pace"],
            "past_gpa": (hist["pass_rate"] or 0.6) * 4.0,
            "recent_gpa": (hist["pass_rate"] or 0.6) * 4.0,
            "avg_difficulty": hist["avg_difficulty"] or 3,
            "credits_taken": hist["courses_completed"] or 0,
        }
        df = pd.get_dummies(pd.DataFrame([row])).reindex(columns=feature_columns, fill_value=0)
        prob = model(torch.tensor(scaler.transform(df), dtype=torch.float32)).item()

        with driver.session(database=settings.DB_NAME) as s:
            prereqs = s.run("""
            MATCH (p:Course)-[:PREREQUISITE_FOR]->(c:Course {id:$cid})
            RETURN p.id AS id, p.name AS name
            """, cid=c["courseId"]).data()
        prereq_str = ", ".join(f"{p['id']} ({p['name']})" for p in prereqs) if prereqs else "None"

        recs.append({
            "courseId": c["courseId"],
            "name": c["name"],
            "credits": int(c["credits"] or 0),
            "dept": c["dept"],
            "difficulty": _difficulty_label(c["difficulty"]),
            "prob": round(prob, 2),
            "label": "High Success" if prob >= .75 else ("Medium Success" if prob >= .5 else "Low Success"),
            "prereqs": prereq_str,
        })

    recs = sorted(recs, key=lambda r: r["prob"], reverse=True)[:max_courses]

    total_required = int(student["totalCredits"])
    credits_completed = int(hist["credits_completed"] or 0)
    current_sem = int(max(1, (hist["courses_completed"] or 0) // (student["courseLoad"] or 3)))

    return {
        "studentId": student_id,
        "degree": student["degreeName"],
        "totalCredits": total_required,
        "creditsCompleted": credits_completed,
        "creditsRemaining": total_required - credits_completed,
        "currentSemester": current_sem,
        "nextSemester": current_sem + 1,
        "targetCredits": int((student["courseLoad"] or 3) * 3),
        "expectedGraduation": str(student["gradDate"]),
        "recommendedCourses": recs,
    }




def get_popular_courses(limit: int = 20, term: str | None = None) -> List[Dict]:
    with driver.session(database=settings.DB_NAME) as s:
        if term:
            # term-specific (requires you ran the term recompute or compute on the fly)
            rows = s.run(
                """
                MATCH (c:Course)
                RETURN c.id AS courseId,
                       c.name AS name,
                       c.department AS dept,
                       c.credits AS credits,
                       c.avgDifficulty AS difficulty,
                       coalesce(c.completions_this_term, 0) AS completions
                ORDER BY completions DESC, name ASC
                LIMIT $limit
                """,
                limit=limit,
            ).data()
        else:
            rows = s.run(
                """
                MATCH (c:Course)
                RETURN c.id AS courseId,
                       c.name AS name,
                       c.department AS dept,
                       c.credits AS credits,
                       c.avgDifficulty AS difficulty,
                       coalesce(c.completions, 0) AS completions
                ORDER BY completions DESC, name ASC
                LIMIT $limit
                """,
                limit=limit,
            ).data()

    # map difficulty number -> label if you store as number
    def diff_label(d):
        if d is None: return "Unknown"
        if d <= 2:    return "Easy"
        if d == 3:    return "Medium"
        return "Hard"

    out = []
    for r in rows:
        r["difficultyLabel"] = diff_label(r.get("difficulty"))
        out.append(r)
    return out
