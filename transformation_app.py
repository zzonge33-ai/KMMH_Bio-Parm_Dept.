import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import io
import plotly.express as px

# 기본 세팅
st.set_page_config(page_title="Transformation AI 파일럿", layout="wide")

# 파일 경로
DATA_PATH = Path("transformation_history.csv")
MODEL_PATH = Path("transformation_model.pkl")

# 기본 컬럼 리스트
COLS = [
    "dna_ng", "temp", "time_sec", "recovery_min", "cells_ug", "colonies",
    "efficiency", "success"
]


# 0. 데이터 로드 또는 초기 생성
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.DataFrame(columns=COLS)
    return df.copy()


def save_data(df):
    df.to_csv(DATA_PATH, index=False)
    # 다시 캐시 업데이트
    load_data.clear()


def is_ready_to_train(df):
    return len(df) >= 10 and "success" in df.columns


# 1. 모델 학습/로드
def get_model_and_scaler(df):
    if not is_ready_to_train(df):
        return None, None

    X = df[["dna_ng", "temp", "time_sec", "recovery_min", "cells_ug"]].values
    y = df["success"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    # 정확도 간단히 표시
    acc = pipe.score(X_test, y_test)

    return pipe, acc


def load_or_train_model(df):
    if MODEL_PATH.exists():
        if is_ready_to_train(df):
            # 데이터가 바뀌었는지 검사
            cached_df_hash = st.session_state.get("df_hash", "")
            cur_hash = hash(df.to_string())
            if cached_df_hash != cur_hash:
                model, acc = get_model_and_scaler(df)
                if model is not None:
                    joblib.dump(model, MODEL_PATH)
                    st.session_state["df_hash"] = cur_hash
            else:
                model = joblib.load(MODEL_PATH)
        else:
            model = None
    else:
        model = None
        if is_ready_to_train(df):
            model, acc = get_model_and_scaler(df)
            if model is not None:
                joblib.dump(model, MODEL_PATH)

    return model


# 2. 탭 1: 실험 입력 + 효율 계산 + 예측
def tab_experiment(df):
    st.header("1. 실험 입력")

    col1, col2 = st.columns(2)

    with col1:
        dna_ng = st.number_input("DNA 양 (ng)", min_value=0.1, value=10.0, step=0.5)
        temp = st.number_input("Heat shock 온도 (°C)", min_value=30.0, value=42.0, step=0.5)
        time_sec = st.number_input("Heat shock 시간 (sec)", min_value=10, value=30, step=5)
        recovery_min = st.number_input("Recovery 시간 (min)", min_value=10, value=30, step=5)
        cells_ug = st.number_input("Competent cell 효율 지표 (cells/μg)", min_value=0.1, value=1.0, step=0.1)

    with col2:
        colonies = st.number_input("관찰된 colony 수", min_value=0, value=100, step=1)

    # 효율 계산
    if colonies > 0 and cells_ug > 0:
        efficiency = colonies / (dna_ng * cells_ug)  # CFU/μg
        efficiency = round(efficiency, 2)
    else:
        efficiency = 0.0

    st.metric("Transformation 효율", f"{efficiency:.2f} CFU/μg")

    # 성공여부 기준
    success_thres = 50  # 예시: 50 colony 이상이면 성공
    success = 1 if colonies >= success_thres else 0

    # 현재 조건
    current = pd.DataFrame([{
        "dna_ng": dna_ng,
        "temp": temp,
        "time_sec": time_sec,
        "recovery_min": recovery_min,
        "cells_ug": cells_ug,
        "colonies": colonies,
        "efficiency": efficiency,
        "success": success
    }])

    # 모델 로드
    model = load_or_train_model(df)

    # 예측
    if model is not None and len(df) >= 10:
        X_new = current[["dna_ng", "temp", "time_sec", "recovery_min", "cells_ug"]].values
        prob = model.predict_proba(X_new)[0, 1]  # 성공 확률
        success_prob = int(prob * 100)
        st.success(f"이 조건의 예측 성공 확률: **{success_prob}%**")
    else:
        st.info("👉 데이터가 부족하거나 아직 모델이 학습되지 않았습니다.")

    # 저장
    if st.button("💾 현재 조건 저장"):
        updated_df = pd.concat([df, current], ignore_index=True)
        save_data(updated_df)
        st.success("데이터가 저장되었습니다.")
        st.session_state.df = updated_df  # 새로고침용
        st.rerun()


# 3. 탭 2: 데이터 업로드 + 그래프
def tab_data_and_graph(df):
    st.header("2. 실험 데이터 업로드 및 그래프")

    # 2‑1 데이터 업로드
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일 업로드 (기존 형식과 동일한 컬럼)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                new_df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                new_df = pd.read_excel(uploaded_file)

            st.info("✅ 파일 읽기 완료")
            st.write("상위 5행")
            st.dataframe(new_df.head(), use_container_width=True)

            # 필요한 컬럼 체크
            essential_cols = ["dna_ng", "temp", "time_sec", "recovery_min", "cells_ug", "colonies"]
            if not all(c in new_df.columns for c in essential_cols):
                st.warning("⚠️ 필수 컬럼이 부족합니다: dna_ng, temp, time_sec, recovery_min, cells_ug, colonies")
            else:
                # efficiency & success 계산
                eps = 1e-8
                new_df["efficiency"] = new_df["colonies"] / (new_df["dna_ng"] * new_df["cells_ug"] + eps)
                success_thres = 50
                new_df["success"] = (new_df["colonies"] >= success_thres).astype(int)

                # 기존 데이터와 합치기
                updated_df = pd.concat([df, new_df], ignore_index=True)
                save_data(updated_df)
                st.session_state.df = updated_df

                st.success("📊 데이터가 반영되었습니다.")
                st.rerun()

        except Exception as e:
            st.error(f"❌ 파일 읽기 실패: {e}")

    # 2‑2 그래프
    if len(df) == 0:
        st.info("데이터가 없으면 그래프를 그릴 수 없습니다.")
        return

    st.subheader("📈 변환 효율 그래프")

    # DNA 양 vs 효율
    fig1 = px.scatter(
        df,
        x="dna_ng", y="efficiency",
        color="success",
        title="DNA 양 vs Transformation 효율 (성공: 1, 실패: 0)",
        labels={"dna_ng": "DNA 양 (ng)", "efficiency": "효율 (CFU/μg)"},
        color_continuous_scale="rdbu"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Heat shock 시간 vs 효율
    fig2 = px.scatter(
        df,
        x="time_sec", y="efficiency",
        color="temp",
        title="Heat shock 시간 vs 효율 (색: 온도)",
        labels={"time_sec": "Heat shock 시간 (sec)", "efficiency": "효율 (CFU/μg)"},
        color_continuous_scale="plasma"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Colony 수 분포
    fig3 = px.histogram(
        df,
        x="colonies",
        nbins=20,
        title="Colony 수 분포",
        labels={"colonies": "Colony 수"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 2‑3 데이터 표
    st.subheader("📋 데이터 테이블")
    st.dataframe(df, use_container_width=True)

    # 2‑4 다운로드
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 전체 데이터 다운로드",
        csv,
        "transformation_data.csv",
        "text/csv"
    )


# 4. 메인 실행
def main():
    st.title("🧫 Transformation AI 파일럿 – 업로드 + 그래프 + 예측 강화")

    # 세션 상태에 데이터 저장
    if "df" not in st.session_state:
        st.session_state.df = load_data()

    df = st.session_state.df

    # 탭 설정
    tab1, tab2 = st.tabs(["실험 입력", "데이터 + 그래프"])

    with tab1:
        tab_experiment(df)

    with tab2:
        tab_data_and_graph(df)

    # 모델 상태 간단히 보여주기
    model = load_or_train_model(df)
    if "success_thres" not in st.session_state:
        st.session_state["success_thres"] = 50

    st.caption("🔗 데이터 저장 위치: `transformation_history.csv`")
    st.caption("🔗 모델 저장 위치: `transformation_model.pkl`")


if __name__ == "__main__":
    main()