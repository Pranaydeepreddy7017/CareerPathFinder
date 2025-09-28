#!/usr/bin/env python3
"""
Neural Network Student Success Prediction (JSON API Ready)
- Connects to Neo4j
- Trains / loads model
- Predicts success for a student in a given course
- Returns JSON instead of print statements
"""

from neo4j import GraphDatabase
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json

# ---------------------------
# STEP 1: Connect to Neo4j
# ---------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Pranay@123"
DB_NAME = "umbc-data"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ---------------------------
# STEP 2: Query Data
# ---------------------------
query = """
MATCH (s:Student)-[c:COMPLETED]->(course:Course)
RETURN 
  s.id AS studentId,
  course.id AS courseId,
  c.grade AS grade,
  c.difficulty AS difficulty,
  c.timeSpent AS timeSpent,
  s.learningStyle AS learningStyle,
  s.preferredCourseLoad AS courseLoad,
  s.preferredPace AS pace
"""

with driver.session(database=DB_NAME) as session:
    result = session.run(query)
    df = pd.DataFrame([r.data() for r in result])

print(f"✅ Data Loaded: {df.shape}")

# ---------------------------
# STEP 3: Feature Engineering
# ---------------------------
grade_map = {"A":4.0,"A-":3.7,"B+":3.3,"B":3.0,"B-":2.7,"C+":2.3,"C":2.0,"C-":1.7,
             "D+":1.3,"D":1.0,"F":0.0,"W":0.0}
df["gpa_points"] = df["grade"].map(grade_map)
df["success"] = (df["gpa_points"] >= 3.0).astype(int)

# Aggregate student history
student_stats = df.groupby("studentId").agg(
    past_gpa=("gpa_points","mean"),
    recent_gpa=("gpa_points", lambda x: x.tail(3).mean()),
    avg_difficulty=("difficulty","mean"),
    credits_taken=("courseId","count")
).reset_index()

df = df.merge(student_stats, on="studentId", how="left")

# Features
X = df[[
    "difficulty", "timeSpent", "learningStyle", "courseLoad", "pace",
    "past_gpa", "recent_gpa", "avg_difficulty", "credits_taken"
]]
y = df["success"]

X = pd.get_dummies(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler + feature columns for reuse
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ---------------------------
# STEP 4: Define NN
# ---------------------------
class StudentNN(nn.Module):
    def __init__(self, input_dim):
        super(StudentNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

model = StudentNN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

MODEL_PATH = "student_success_nn.pth"
TRAIN_MODEL = True  # set to False to load pre-trained

# ---------------------------
# STEP 5: Train or Load
# ---------------------------
if TRAIN_MODEL:
    print("\n🚀 Training Neural Network...")
    for epoch in range(20):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
else:
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"✅ Loaded pre-trained model from {MODEL_PATH}")

# ---------------------------
# STEP 6: Predict Function (returns JSON)
# ---------------------------
def predict_student_course(student_id, course_id):
    with driver.session(database=DB_NAME) as session:
        student_data = session.run("""
        MATCH (s:Student {id: $sid})-[:COMPLETED]->(c:Course)
        RETURN avg(c.difficulty) AS avg_difficulty,
               avg(CASE WHEN c.grade IN ['A','A-','B+','B'] THEN 1 ELSE 0 END) AS pass_rate,
               avg(c.timeSpent) AS avg_time,
               s.learningStyle AS learningStyle,
               s.preferredCourseLoad AS courseLoad,
               s.preferredPace AS pace
        """, sid=student_id).single()

        course_data = session.run("""
        MATCH (c:Course {id: $cid})
        RETURN c.level AS difficulty
        """, cid=course_id).single()

    if not student_data or not course_data:
        return {"error": f"Missing data for {student_id} or {course_id}"}

    row = {
        "difficulty": course_data["difficulty"] or 3,
        "timeSpent": student_data["avg_time"] or 6,
        "learningStyle": student_data["learningStyle"],
        "courseLoad": student_data["courseLoad"],
        "pace": student_data["pace"],
        "past_gpa": student_data["pass_rate"] * 4.0,
        "recent_gpa": student_data["pass_rate"] * 4.0,
        "avg_difficulty": student_data["avg_difficulty"] or 3,
        "credits_taken": 10
    }

    df_pred = pd.DataFrame([row])
    df_pred = pd.get_dummies(df_pred).reindex(columns=X.columns, fill_value=0)
    X_pred = scaler.transform(df_pred)

    with torch.no_grad():
        prob = model(torch.tensor(X_pred, dtype=torch.float32)).item()

    result = "Likely to Succeed" if prob >= 0.5 else "At Risk of Failing"

    return {
        "student_id": student_id,
        "course_id": course_id,
        "probability": round(prob, 2),
        "result": result
    }

# ---------------------------
# Example Run
# ---------------------------
if __name__ == "__main__":
    prediction = predict_student_course("RQ34157", "BUUU 200-6")
    print(json.dumps(prediction, indent=2))



import joblib

# Save scaler + feature columns for later use
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
torch.save(model.state_dict(), "student_success_nn.pth")
print("✅ Saved scaler, feature columns, and NN model")


#!/usr/bin/env python3
"""
Course Recommendation System with Degree Progress Summary (JSON output ready)
- Returns degree progress, completed courses, and recommended courses
- Success labels: High / Medium / Low
- Difficulty shown as info column
- Prerequisites included for recommendations
"""

from neo4j import GraphDatabase
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import json

# ---------------------------
# STEP 1: Connect to Neo4j
# ---------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Pranay@123"
DB_NAME = "umbc-data"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ---------------------------
# STEP 2: Load trained model + scaler
# ---------------------------
MODEL_PATH = "student_success_nn.pth"
SCALER_PATH = "scaler.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"

scaler: StandardScaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

class StudentNN(nn.Module):
    def __init__(self, input_dim):
        super(StudentNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

model = StudentNN(len(feature_columns))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ---------------------------
# STEP 3: Utility to clean Neo4j records
# ---------------------------
def clean_record(record):
    clean = {}
    for k, v in record.items():
        if hasattr(v, "to_native"):   # Neo4j numeric
            clean[k] = v.to_native()
        elif hasattr(v, "isoformat"): # Neo4j date/datetime
            clean[k] = v.isoformat()
        else:
            clean[k] = v
    return clean

# ---------------------------
# STEP 4: Difficulty Mapping
# ---------------------------
def map_difficulty(val):
    if val is None:
        return "Unknown"
    elif val <= 2:
        return "Easy"
    elif val == 3:
        return "Medium"
    else:
        return "Hard"

# ---------------------------
# STEP 5: Recommendation function
# ---------------------------
def recommend_courses(student_id, term="Fall2025", allow_cross_dept=False, max_courses=10):
    with driver.session(database=DB_NAME) as session:
        # Student & degree info
        student = session.run("""
        MATCH (s:Student {id:$sid})-[:PURSUING]->(d:Degree)
        RETURN s.id AS id, s.learningStyle AS learningStyle,
               s.preferredCourseLoad AS courseLoad, s.preferredPace AS pace,
               s.enrollmentDate AS enrollmentDate, s.expectedGraduation AS gradDate,
               d.name AS degreeName, d.id AS degreeId,
               d.totalCreditsRequired AS totalCredits
        """, sid=student_id).single()

        if not student:
            return {"error": f"No student {student_id}"}

        # Student history
        history = session.run("""
        MATCH (s:Student {id:$sid})-[c:COMPLETED]->(course:Course)
        RETURN avg(c.difficulty) AS avg_difficulty,
               avg(CASE WHEN c.grade IN ['A','A-','B+','B'] THEN 1.0 ELSE 0 END) AS pass_rate,
               avg(c.timeSpent) AS avg_time,
               sum(course.credits) AS credits_completed,
               count(c) AS courses_completed
        """, sid=student_id).single()

        credits_completed = history["credits_completed"] or 0
        total_required = student["totalCredits"]
        credits_remaining = total_required - credits_completed
        current_sem = max(1, history["courses_completed"] // student["courseLoad"])
        target_sem = current_sem + 1

        # Completed courses
        completed = session.run("""
        MATCH (s:Student {id:$sid})-[c:COMPLETED]->(course:Course)
        RETURN course.id AS courseId, course.name AS name,
               course.credits AS credits, course.department AS dept
        ORDER BY course.id
        """, sid=student_id).data()
        completed = [clean_record(r) for r in completed]

        # Courses offered for next term
        if allow_cross_dept:
            course_query = """
            MATCH (c:Course)-[:OFFERED_IN]->(t:Term {id:$term})
            WHERE NOT EXISTS { MATCH (:Student {id:$sid})-[:COMPLETED]->(c) }
            RETURN c.id AS courseId, c.name AS name,
                   c.credits AS credits, c.avgDifficulty AS difficulty,
                   c.department AS dept
            """
        else:
            course_query = """
            MATCH (s:Student {id:$sid})-[:PURSUING]->(d:Degree)<-[:PART_OF]-(r:RequirementGroup)<-[:FULFILLS]-(c:Course)-[:OFFERED_IN]->(t:Term {id:$term})
            WHERE NOT EXISTS { MATCH (:Student {id:$sid})-[:COMPLETED]->(c) }
            RETURN DISTINCT c.id AS courseId, c.name AS name,
                   c.credits AS credits, c.avgDifficulty AS difficulty,
                   c.department AS dept
            """
        courses = session.run(course_query, sid=student_id, term=term).data()

    if not courses:
        return {"error": f"No available courses for {student_id} in {term}"}

    recs = []
    for c in courses:
        row = {
            "difficulty": c["difficulty"] or 3,
            "timeSpent": history["avg_time"] or 6,
            "learningStyle": student["learningStyle"],
            "courseLoad": student["courseLoad"],
            "pace": student["pace"],
            "past_gpa": history["pass_rate"] * 4.0 if history["pass_rate"] else 2.5,
            "recent_gpa": history["pass_rate"] * 4.0 if history["pass_rate"] else 2.5,
            "avg_difficulty": history["avg_difficulty"] or 3,
            "credits_taken": history["courses_completed"] or 0
        }
        df_pred = pd.DataFrame([row])
        df_pred = pd.get_dummies(df_pred).reindex(columns=feature_columns, fill_value=0)
        X_pred = scaler.transform(df_pred)
        prob = model(torch.tensor(X_pred, dtype=torch.float32)).item()

        # Label by probability
        if prob >= 0.75:
            label = "High Success"
        elif prob >= 0.50:
            label = "Medium Success"
        else:
            label = "Low Success"

        with driver.session(database=DB_NAME) as session:
            prereqs = session.run("""
            MATCH (p:Course)-[:PREREQUISITE_FOR]->(c:Course {id:$cid})
            RETURN p.id AS prereqId, p.name AS prereqName
            """, cid=c["courseId"]).data()
        prereq_str = ", ".join([f"{p['prereqId']} ({p['prereqName']})" for p in prereqs]) if prereqs else "None"

        recs.append({
            "courseId": c["courseId"], "name": c["name"],
            "credits": int(c["credits"]) if c["credits"] else 0,
            "dept": c["dept"],
            "difficulty": map_difficulty(c["difficulty"]),
            "prob": round(prob, 2),
            "label": label,
            "prereqs": prereq_str
        })

    recs = sorted(recs, key=lambda r: r["prob"], reverse=True)[:max_courses]

    return {
        "studentId": student_id,
        "degree": student["degreeName"],
        "totalCredits": int(total_required),
        "creditsCompleted": int(credits_completed),
        "creditsRemaining": int(credits_remaining),
        "currentSemester": int(current_sem),
        "nextSemester": int(target_sem),
        "targetCredits": int(student["courseLoad"] * 3),
        "expectedGraduation": (
            student["gradDate"].isoformat() if hasattr(student["gradDate"], "isoformat") else str(student["gradDate"])
        ),
        "completedCourses": completed,
        "recommendedCourses": recs
    }

# recommend.py
# recommend.py
# recommend.py
from typing import Dict, List, Optional
from core import get_driver, settings

def popular_courses(limit: int = 10, term: Optional[str] = None) -> Dict:
    driver = get_driver()
    with driver.session(database=settings.DB_NAME) as s:
        if term:
            # Popular among courses offered in a specific term
            cypher = """
            MATCH (:Student)-[:COMPLETED]->(c:Course)-[:OFFERED_IN]->(t:Term {id:$term})
            RETURN c.id AS courseId, c.name AS name, c.department AS dept,
                   count(*) AS completions
            ORDER BY completions DESC, courseId ASC
            LIMIT $limit
            """
            rows = s.run(cypher, term=term, limit=limit).data()
        else:
            # Global popularity
            cypher = """
            MATCH (:Student)-[:COMPLETED]->(c:Course)
            RETURN c.id AS courseId, c.name AS name, c.department AS dept,
                   count(*) AS completions
            ORDER BY completions DESC, courseId ASC
            LIMIT $limit
            """
            rows = s.run(cypher, limit=limit).data()

    return {"popular": [
        {"courseId": r["courseId"], "name": r["name"], "dept": r["dept"], "completions": int(r["completions"])}
        for r in rows
    ]}


def similar_students(student_id: str, top_k: int = 5) -> Dict:
    driver = get_driver()
    with driver.session(database=settings.DB_NAME) as s:
        rows = s.run("""
        MATCH (me:Student {id:$sid})-[:COMPLETED]->(c:Course)<-[:COMPLETED]-(peer:Student)
        WHERE peer.id <> $sid
        WITH peer, count(DISTINCT c) AS overlap
        ORDER BY overlap DESC, peer.id ASC
        LIMIT $top_k
        RETURN peer.id AS peerId, overlap
        """, sid=student_id, top_k=top_k).data()
    return {"studentId": student_id, "peers": [{"studentId": r["peerId"], "overlap": int(r["overlap"])} for r in rows]}



# ---------------------------
# Example Run
# ---------------------------
# if __name__ == "__main__":
#     result = recommend_courses("RQ34157", term="Spring2024", allow_cross_dept=False, max_courses=12)
#     print(json.dumps(result, indent=2, ensure_ascii=False))