import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import io

st.set_page_config(page_title="Transformation Pilot", page_icon="🧫", layout="wide")
st.title("🧫 Transformation 실험 파일럿 v3")
st.caption("직접 입력 + 데이터 업로드 + AI 예측")

DATA_FILE = Path("transformation_history.csv")
MODEL_FILE = Path("transformation_model.pkl")

def make_demo_data(n=120):
    rng = np.random.default_rng(42)
    dna_ng = rng.choice([5, 10, 20, 50], size=n)
    temp = rng.choice([37, 42, 45], size=n)
    time_sec = rng.choice([15, 30, 60], size=n)
    recovery_min = rng.choice([20, 30, 60], size=n)
    cells_ug = rng.choice([0.5, 1.0, 2.0], size=n)
    colonies = (
        20 + dna_ng * 6 + (temp == 42) * 18 + (time_sec == 30) * 10
        + (recovery_min == 30) * 8 + (cells_ug * 12) + rng.normal(0, 12, n)
    )
    colonies = np.clip(colonies, 0, None).astype(int)
    efficiency = colonies / (dna_ng / 1000)
    success = (colonies >= 80).astype(int)
    return pd.DataFrame({
        "dna_ng": dna_ng, "temp": temp, "time_sec": time_sec,
        "recovery_min": recovery_min, "cells_ug": cells_ug,
        "colonies": colonies, "efficiency": efficiency, "success": success
    })

def load_data():
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    df = make_demo_data()
    df.to_csv(DATA_FILE, index=False)
    return df

def train_model(df):
    if len(df) < 10 or df["success"].nunique() < 2:
        return None
    X = df[["dna_ng", "temp", "time_sec", "recovery_min", "cells_ug", "efficiency"]]
    y = df["success"]
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return 