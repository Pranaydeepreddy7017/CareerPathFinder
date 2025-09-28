# api.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# import your service functions
from service import predict_student_course, recommend_courses
from career import get_career_paths
from recommend import popular_courses, similar_students
from typing import Optional

app = FastAPI(title="PathFinder API", version="1.0")

# Enable CORS (for Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Models
# --------------------------
class LoginRequest(BaseModel):
    name: str
    student_id: str

# --------------------------
# Routes
# --------------------------

@app.get("/")
def root():
    return {"ok": True}

# 🔑 Login endpoint
@app.post("/login")
def login(payload: LoginRequest):
    """
    Accepts student name + ID.
    Returns student profile stub.
    (Future: you can add DB validation here.)
    """
    return {
        "student_id": payload.student_id,
        "name": payload.name,
        "message": "Login successful"
    }

@app.get("/predict")
def predict(student_id: str = Query(...), course_id: str = Query(...)):
    return predict_student_course(student_id, course_id)

@app.get("/recommend")
def recommend(student_id: str = Query(...),
              term: str = Query("Spring2024"),
              max_courses: int = Query(12),
              allow_cross_dept: bool = Query(False)):
    return recommend_courses(student_id, term=term,
                             allow_cross_dept=allow_cross_dept,
                             max_courses=max_courses)

@app.get("/career")
def career(student_id: str = Query(...)):
    return get_career_paths(student_id)

@app.get("/popular-courses")
def popular(limit: int = 10, term: Optional[str] = None):
    return popular_courses(limit=limit, term=term)

@app.get("/similar-students")
def similar(student_id: str, top_k: int = 5):
    return similar_students(student_id=student_id, top_k=top_k)

@app.get("/health/env")
def health_env():
    from core import settings
    return {
        "neo4j_uri": settings.NEO4J_URI,
        "db": settings.DB_NAME,
        "gemini_key_loaded": bool(settings.GEMINI_API_KEY),
    }
