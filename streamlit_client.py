"""Streamlit dashboard that consumes the Dockerized MLflow model server.

Launch:
    streamlit run streamlit_client.py

Requires the model container to be running on http://127.0.0.1:5001
(see the Docker section of mlflow.ipynb).
"""

from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = os.environ.get("MODEL_API_URL", "http://127.0.0.1:5001")
FEATURE_COLUMNS = [
    "pregnancies",
    "glucose",
    "diastolic",
    "triceps",
    "insulin",
    "bmi",
    "dpf",
    "age",
]
SAMPLE_PATIENT = {
    "pregnancies": 2,
    "glucose": 138,
    "diastolic": 62,
    "triceps": 35,
    "insulin": 0,
    "bmi": 33.6,
    "dpf": 0.127,
    "age": 47,
}
FEATURE_HELP = {
    "pregnancies": "Number of times pregnant",
    "glucose": "Plasma glucose concentration (2-hour OGTT, mg/dL)",
    "diastolic": "Diastolic blood pressure (mm Hg)",
    "triceps": "Triceps skin fold thickness (mm)",
    "insulin": "2-Hour serum insulin (mu U/ml)",
    "bmi": "Body mass index (weight kg / height m^2)",
    "dpf": "Diabetes pedigree function (family history score)",
    "age": "Age (years)",
}


def check_health(api_url: str) -> bool:
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def call_model_api(records: list[dict[str, Any]], api_url: str) -> list[int]:
    """POST records to MLflow's /invocations endpoint and return predictions."""
    response = requests.post(
        f"{api_url}/invocations",
        json={"dataframe_records": records},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["predictions"]


def render_sidebar() -> str:
    st.sidebar.header("Configuration")
    api_url = st.sidebar.text_input(
        "Model API URL",
        value=DEFAULT_API_URL,
        help="Set MODEL_API_URL env var to change the default.",
    )

    healthy = check_health(api_url)
    if healthy:
        st.sidebar.success("API is healthy")
    else:
        st.sidebar.error("API unreachable — is the container running?")
        st.sidebar.code(
            "docker ps --filter name=pima-diabetes-server",
            language="bash",
        )

    with st.sidebar.expander("Heads up: MLflow scoring server"):
        st.write(
            "The default `/invocations` endpoint only returns class **labels** "
            "(0/1), not probabilities. To expose `predict_proba`, you'd wrap "
            "the model in a custom `mlflow.pyfunc.PythonModel`."
        )

    with st.sidebar.expander("curl equivalent"):
        st.code(
            f'curl -X POST {api_url}/invocations \\\n'
            '  -H "Content-Type: application/json" \\\n'
            "  -d '{\"dataframe_records\": [{\"pregnancies\": 2, ...}]}'",
            language="bash",
        )

    return api_url


def render_single_prediction_tab(api_url: str) -> None:
    st.subheader("Single patient prediction")
    with st.form("single_prediction"):
        cols = st.columns(2)
        inputs: dict[str, float] = {}
        for index, feature in enumerate(FEATURE_COLUMNS):
            with cols[index % 2]:
                inputs[feature] = st.number_input(
                    feature,
                    value=float(SAMPLE_PATIENT[feature]),
                    help=FEATURE_HELP[feature],
                    format="%.3f",
                )
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    try:
        predictions = call_model_api([inputs], api_url)
    except requests.RequestException as error:
        st.error(f"Could not reach the model API: {error}")
        return
    except (KeyError, ValueError) as error:
        st.error(f"Unexpected response from the API: {error}")
        return

    label = int(predictions[0])
    if label == 1:
        st.error("Prediction: Diabetic")
    else:
        st.success("Prediction: Not Diabetic")
    st.json({"request": inputs, "response": {"predictions": predictions}})


def render_batch_prediction_tab(api_url: str) -> None:
    st.subheader("Batch prediction from CSV")
    st.caption(
        f"Upload a CSV with columns: {', '.join(FEATURE_COLUMNS)}"
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        return

    df = pd.read_csv(uploaded)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    records = df[FEATURE_COLUMNS].to_dict(orient="records")
    try:
        predictions = call_model_api(records, api_url)
    except requests.RequestException as error:
        st.error(f"Could not reach the model API: {error}")
        return

    result = df.copy()
    result["prediction"] = predictions
    result["label"] = result["prediction"].map({0: "Not Diabetic", 1: "Diabetic"})
    st.dataframe(result, use_container_width=True)

    buffer = io.StringIO()
    result.to_csv(buffer, index=False)
    st.download_button(
        "Download predictions CSV",
        data=buffer.getvalue(),
        file_name="predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="Pima Diabetes — MLflow Client",
        page_icon="🩺",
        layout="wide",
    )
    st.title("Pima Diabetes — MLflow Model Client")
    st.caption(
        "Calls the Dockerized champion model registered as "
        "`pima_diabetes_rf_classifier@champion`."
    )

    api_url = render_sidebar()
    single_tab, batch_tab = st.tabs(["Single patient", "Batch CSV"])
    with single_tab:
        render_single_prediction_tab(api_url)
    with batch_tab:
        render_batch_prediction_tab(api_url)


if __name__ == "__main__":
    main()
