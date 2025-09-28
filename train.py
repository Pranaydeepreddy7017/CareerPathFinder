# core.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import joblib
import torch
import torch.nn as nn

load_dotenv()

# ---------- Model ----------
class StudentNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# ---------- Config ----------
@dataclass
class Settings:
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASS: str = os.getenv("NEO4J_PASS", "neo4j")
    DB_NAME:   str = os.getenv("DB_NAME", "umbc-data")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "student_success_nn.pth")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "scaler.pkl")
    FEATURES_PATH: str = os.getenv("FEATURE_COLUMNS_PATH", "feature_columns.pkl")

settings = Settings()

# ---------- Artifacts ----------
def load_artifacts():
    feature_columns = joblib.load(settings.FEATURES_PATH)
    scaler = joblib.load(settings.SCALER_PATH)
    model = StudentNN(len(feature_columns))
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, scaler, feature_columns


if __name__ == "__main__":
    train()  # whatever your train function is
