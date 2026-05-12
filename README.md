---
title: Pima Diabetes MLflow Demo
emoji: 🩺
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Pima Diabetes — MLflow + Streamlit Demo

A teaching demo showing the full MLOps loop on the classic Pima Indians Diabetes dataset:

1. **Train & track** — RandomForest + XGBoost runs logged to MLflow with grid-searched hyperparameters
2. **Register** — Champion model promoted via the MLflow Model Registry alias `@champion`
3. **Serve** — `mlflow models serve` exposes a REST API at `/invocations`
4. **Consume** — Streamlit UI wraps the API with a clickable form

This Space runs **both services in one container**:

- MLflow scoring server on internal port `5001`
- Streamlit UI on port `7860` (the public URL)

## Using the UI

The **Single patient** tab predicts diabetes risk for one record using the 8 standard features. The **Batch CSV** tab uploads a CSV and returns predictions as a downloadable file.

> The default MLflow scoring endpoint returns class labels (0/1), not probabilities. To expose `predict_proba` you'd wrap the model in a custom `mlflow.pyfunc.PythonModel`.

## Run locally

```bash
git clone https://github.com/SubaashNair/pima-diabetes-mlflow
cd pima-diabetes-mlflow

# Build and run (both ports exposed locally)
docker build -t pima-demo .
docker run --rm -p 7860:7860 -p 5001:5001 pima-demo
```

Open <http://localhost:7860> for the UI, and `curl http://localhost:5001/invocations` works directly for the API.

### Example API call

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"pregnancies":2,"glucose":138,"diastolic":62,"triceps":35,"insulin":0,"bmi":33.6,"dpf":0.127,"age":47}]}'
```

Response:

```json
{"predictions": [1]}
```

## Files

- `mlflow.ipynb` — the full training & registry workflow
- `streamlit_client.py` — Streamlit UI (env-driven via `MODEL_API_URL`)
- `model/` — the bundled champion artifact (XGBoost)
- `Dockerfile` + `start.sh` — combined container that runs both services
- `docker.md` — extended notes on the manual `mlflow models build-docker` workflow

## Notes

- HF Spaces free CPU tier: first request after idle may take 30–60 seconds (cold start).
- The bundled `model/` is `pima_diabetes_rf_classifier` version 2 (XGBoost, not RF — XGBoost beat RF in the comparison).
