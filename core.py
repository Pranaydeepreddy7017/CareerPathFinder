# core.py
from functools import lru_cache
from typing import Tuple, List
import joblib
import torch
from pydantic_settings import BaseSettings
from neo4j import GraphDatabase

# ---------- Settings ----------
class Settings(BaseSettings):
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASS: str
    DB_NAME: str
    GEMINI_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

# ---------- Neo4j Driver ----------
@lru_cache
def get_driver():
    return GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASS))

# ---------- Artifacts ----------
def load_artifacts() -> Tuple[torch.nn.Module, object, List[str]]:
    """
    Loads model, scaler, and feature_columns that your training step saved:
    - student_success_nn.pth
    - scaler.pkl
    - feature_columns.pkl
    """
    import torch.nn as nn

    feature_columns = joblib.load("feature_columns.pkl")
    scaler = joblib.load("scaler.pkl")

    class StudentNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.layers(x)

    model = StudentNN(len(feature_columns))
    model.load_state_dict(torch.load("student_success_nn.pth", map_location="cpu"))
    model.eval()

    return model, scaler, feature_columns
