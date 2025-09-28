# career.py
from __future__ import annotations
from typing import Dict, List, Optional
import json
import os

import google.generativeai as genai
from core import settings, get_driver

# -----------------------------
# Gemini setup (with safe fallback)
# -----------------------------
MODEL_CANDIDATES = [
    # try these in order; we'll fall back to the last one if needed
    "gemini-flash-latest",          # Points to the latest flash model
    "gemini-pro-latest",            # Points to the latest pro model
    "gemini-2.5-flash",             # Specific 2.5 Flash model
    "gemini-2.5-pro",
]

_GEMINI_READY = False
_MODEL_ID = None

def _init_gemini() -> None:
    global _GEMINI_READY, _MODEL_ID
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        _GEMINI_READY = False
        return
    try:
        genai.configure(api_key=api_key)
        # pick first constructible model
        for mid in MODEL_CANDIDATES:
            try:
                _ = genai.GenerativeModel(mid)
                _MODEL_ID = mid
                _GEMINI_READY = True
                return
            except Exception:
                continue
        # last resort: still try to use the last candidate
        _MODEL_ID = MODEL_CANDIDATES[-1]
        _ = genai.GenerativeModel(_MODEL_ID)
        _GEMINI_READY = True
    except Exception:
        _GEMINI_READY = False
        _MODEL_ID = None

_init_gemini()

# -----------------------------
# Helpers
# -----------------------------
def _confidence_label(pct: int) -> str:
    if pct >= 75:
        return "High"
    if pct >= 50:
        return "Medium"
    return "Low"

def _postprocess_paths(raw_paths) -> List[Dict]:
    """Clamp probability to 0–1, convert to %, sort, cap to 3, add confidence."""
    if not isinstance(raw_paths, list):
        return []
    cleaned = []
    for p in raw_paths:
        title = (p.get("title") or "Career Path").strip()
        prob = p.get("probability", 0.5)
        try:
            prob = float(prob)
        except Exception:
            prob = 0.5
        prob = max(0.0, min(1.0, prob))
        pct = round(prob * 100)
        cleaned.append({
            "title": title,
            "probability": pct,
            "confidence": _confidence_label(pct),
            "why": p.get("why") or "",
            "recommendedNextCourses": p.get("recommendedNextCourses") or [],
        })
    cleaned.sort(key=lambda x: x["probability"], reverse=True)
    return cleaned[:3]

def _snap_to_catalog(course_suggestions: List[str]) -> List[str]:
    """Map free-text course ideas to real course IDs from Neo4j (best-effort)."""
    if not course_suggestions:
        return []
    with get_driver().session(database=settings.DB_NAME) as s:
        rows = s.run("""
            MATCH (c:Course)
            RETURN c.id AS id, c.name AS name, c.level AS lvl, c.department AS dept
        """).data()
    # VERY light heuristic match (fast enough for hackathon)
    snapped: List[str] = []
    for sug in course_suggestions:
        sug_low = (sug or "").lower()
        best = None
        best_score = 0.0
        for r in rows:
            score = 0.0
            name = (r["name"] or "").lower()
            if not name:
                continue
            # token overlap
            for tok in sug_low.split():
                if tok and tok in name:
                    score += 1.0
            # level hint
            if r.get("lvl") and str(r["lvl"]) in sug_low:
                score += 0.5
            # dept hint (first 2 letters of first word)
            dept_hint = (r.get("dept") or "").lower().split(" ")[0][:2]
            if dept_hint and dept_hint in sug_low:
                score += 0.25
            if score > best_score:
                best_score, best = score, r["id"]
        snapped.append(best or sug)  # fallback to original text
    # de-dupe preserving order
    seen, out = set(), []
    for c in snapped:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def _filter_to_valid_next_term(course_ids: List[str], term_id: Optional[str]) -> List[str]:
    """Keep only courses actually offered in term_id (if provided)."""
    if not course_ids or not term_id:
        return course_ids or []
    with get_driver().session(database=settings.DB_NAME) as s:
        rows = s.run("""
            UNWIND $ids AS cid
            MATCH (c:Course {id: cid})-[:OFFERED_IN]->(t:Term {id:$term})
            RETURN c.id AS id
        """, ids=course_ids, term=term_id).data()
    offered = {r["id"] for r in rows}
    return [c for c in course_ids if c in offered]

def _fallback_paths() -> List[Dict]:
    """Deterministic fallback if LLM is unavailable."""
    return [
        {
            "title": "Software Engineer",
            "probability": 72,
            "confidence": _confidence_label(72),
            "why": "General tech-oriented path with broad applicability.",
            "recommendedNextCourses": ["CSAA 200", "CSJJ 300"]
        },
        {
            "title": "Data Engineer",
            "probability": 58,
            "confidence": _confidence_label(58),
            "why": "Data systems + analytics exposure likely beneficial.",
            "recommendedNextCourses": ["CSDB 300", "CSBD 300"]
        },
        {
            "title": "Bioinformatics Analyst",
            "probability": 49,
            "confidence": _confidence_label(49),
            "why": "Mix of biology and computing can translate well.",
            "recommendedNextCourses": ["BINF 300", "BGEN 300"]
        },
    ]

# -----------------------------
# Public API
# -----------------------------
def get_career_paths(student_id: str, term: Optional[str] = None) -> Dict:
    """
    Build a compact student profile from Neo4j, ask Gemini for 2–3 career paths,
    snap course suggestions to your catalog, validate against next-term offerings,
    and return UI-ready JSON.
    """
    # 1) Pull a tiny profile for prompting
    try:
        with get_driver().session(database=settings.DB_NAME) as s:
            taken_rows = s.run("""
                MATCH (:Student {id:$sid})-[c:COMPLETED]->(course:Course)
                RETURN course.id AS id, course.name AS name, course.department AS dept
                ORDER BY course.level, course.id
                LIMIT 30
            """, sid=student_id).data()

            deg = s.run("""
                MATCH (:Student {id:$sid})-[:PURSUING]->(d:Degree)
                RETURN d.name AS degreeName
                LIMIT 1
            """, sid=student_id).single()
    except Exception as e:
        # Neo4j error — return a safe response with message
        return {
            "studentId": student_id,
            "degree": "Unknown Degree",
            "paths": _fallback_paths(),
            "error": f"Neo4j error: {e}",
        }

    degree_name = (deg["degreeName"] if deg and deg.get("degreeName") else "Undeclared")
    course_summary = [{"id": r["id"], "name": r["name"], "dept": r["dept"]} for r in taken_rows]

    # 2) If Gemini unavailable, return fallback immediately
    if not _GEMINI_READY or not _MODEL_ID:
        return {
            "studentId": student_id,
            "degree": degree_name,
            "paths": _fallback_paths(),
            "model": None,
            "llm_error": "Gemini not configured or unavailable"
        }

    # 3) Build prompt (STRICT JSON)
    prompt = f"""
You are an academic/career guide. Based on the student's degree and past courses,
suggest 2–3 plausible tech/biology career paths with a probability (0–1) and
2–4 “next course” suggestions that look like this catalog style (e.g., “CSAA 200”).

Return STRICT JSON only in this exact shape — no extra text:

{{
  "studentId": "{student_id}",
  "degree": "{degree_name}",
  "paths": [
    {{
      "title": "Software Engineer",
      "probability": 0.75,
      "why": "Short reason tied to prior courses.",
      "recommendedNextCourses": ["CSAA 200", "CSJJ 300"]
    }},
    {{
      "title": "Network Engineer",
      "probability": 0.5,
      "why": "Short reason tied to prior courses.",
      "recommendedNextCourses": ["CSNN 100", "CSRR 200-8"]
    }}
  ]
}}

Student courses (sample):
{json.dumps(course_summary, ensure_ascii=False)}
"""

    # 4) Call Gemini and parse JSON robustly
    try:
        model = genai.GenerativeModel(_MODEL_ID)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        # attempt to slice JSON if extra tokens
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]
        data = json.loads(text)
    except Exception as e:
        # LLM failure: still succeed with fallback paths (but keep error for visibility)
        return {
            "studentId": student_id,
            "degree": degree_name,
            "paths": _fallback_paths(),
            "model": _MODEL_ID,
            "llm_error": str(e)
        }

    # 5) Post-process (probability -> %, confidence), snap courses, filter by term
    data.setdefault("studentId", student_id)
    data.setdefault("degree", degree_name)
    paths = _postprocess_paths(data.get("paths", []))

    # Snap + filter
    for p in paths:
        snapped = _snap_to_catalog(p.get("recommendedNextCourses", []))
        p["recommendedNextCourses"] = _filter_to_valid_next_term(snapped, term)

    return {
        "studentId": data.get("studentId", student_id),
        "degree": data.get("degree", degree_name),
        "paths": paths,
        "model": _MODEL_ID
    }
