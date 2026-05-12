# End-to-End MLflow Project: Train, Track, Serve, Deploy

A complete walkthrough of the Pima Diabetes MLflow project — from training the first model to a publicly hosted live demo on Hugging Face Spaces, with everything in between.

This document is the **how and the why** of every step. If you only want commands, skip to the code blocks. If you want to actually learn what is happening, read the prose.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites](#2-prerequisites)
3. [Part A — MLflow Tracking](#part-a--mlflow-tracking)
4. [Part B — Training and Comparing Models](#part-b--training-and-comparing-models)
5. [Part C — Model Registry and Aliases](#part-c--model-registry-and-aliases)
6. [Part D — Validating the Model Before Serving](#part-d--validating-the-model-before-serving)
7. [Part E — Serving the Model as a REST API](#part-e--serving-the-model-as-a-rest-api)
8. [Part F — Containerizing with Docker (auto-generated)](#part-f--containerizing-with-docker-auto-generated)
9. [Part G — Custom Combined Dockerfile (Streamlit + MLflow API)](#part-g--custom-combined-dockerfile-streamlit--mlflow-api)
10. [Part H — Streamlit Frontend](#part-h--streamlit-frontend)
11. [Part I — Pushing to GitHub](#part-i--pushing-to-github)
12. [Part J — Deploying to Hugging Face Spaces](#part-j--deploying-to-hugging-face-spaces)
13. [Part K — CI/CD: GitHub Actions Sync to HF](#part-k--cicd-github-actions-sync-to-hf)
14. [Part L — Why this app won't run on Streamlit Community Cloud as-is](#part-l--why-this-app-wont-run-on-streamlit-community-cloud-as-is)
15. [Troubleshooting](#troubleshooting)
16. [Glossary](#glossary)

---

## 1. Project Overview

**What we built:** A diabetes-risk classifier trained on the Pima Indians dataset, served as a REST API and a Streamlit web UI, packaged in one Docker container, hosted publicly on Hugging Face Spaces, and synced from GitHub via a CI workflow.

**The full pipeline:**

```
Train models (RF + XGBoost)
    ↓
Log every run to MLflow (params, metrics, artifacts)
    ↓
Compare runs in the MLflow UI
    ↓
Register the best model with an alias (@champion)
    ↓
Validate locally
    ↓
Serve as REST API (mlflow models serve)
    ↓
Containerize (Docker)
    ↓
Push to GitHub (source) + HF Space (hosted demo)
    ↓
CI auto-syncs GitHub → HF on every commit
```

**Tech stack:**

| Layer | Tool |
|---|---|
| Tracking | MLflow 3.12 |
| Models | scikit-learn (RandomForest) + XGBoost |
| Data | Pima Indians Diabetes (public CSV) |
| API | MLflow scoring server (FastAPI under the hood) |
| UI | Streamlit |
| Containers | Docker (python:3.12-slim base) |
| Hosting | Hugging Face Spaces (Docker SDK) |
| CI/CD | GitHub Actions |

---

## 2. Prerequisites

Install once, then everything in this guide works:

```bash
# Python deps
pip install mlflow xgboost scikit-learn pandas streamlit requests

# CLI tools
brew install gh git-lfs                 # macOS
# OR: sudo apt-get install gh git-lfs   # Ubuntu/Debian

git lfs install
```

Accounts you need:

- **GitHub** account (with `gh auth login`)
- **Hugging Face** account (free) with a write-scope token from https://huggingface.co/settings/tokens
- **Docker Desktop** installed and running

Verify:

```bash
mlflow --version          # → mlflow, version 3.12.x
docker --version          # → Docker version 28.x
docker info               # → must succeed (means daemon is running)
gh auth status            # → must show "Logged in"
git lfs version           # → git-lfs/3.x
```

---

## Part A — MLflow Tracking

### A1. Concepts

**Experiment:** a named container for related runs. Think "this notebook" or "this hyperparameter sweep."

**Run:** one execution of model training. Has parameters, metrics, tags, and artifacts (the trained model file).

**Tracking server:** an HTTP service that stores runs. Backed by SQLite locally, Postgres/MySQL in production. Default URL: `http://127.0.0.1:5000`.

**Backend store:** where the run *metadata* lives (SQLite by default).

**Artifact store:** where the run *files* live (a directory on disk by default).

### A2. Starting the tracking server

The very first command you run before any logging code:

```bash
mlflow ui
```

This:
- Listens on `http://127.0.0.1:5000`
- Uses `sqlite:///mlflow.db` as the backend (creates `mlflow.db` in your CWD)
- Uses `./mlruns` as the artifact store

Leave this running in a dedicated terminal tab. **Closing it = no tracking.**

To stop it later:

```bash
# Find the process holding port 5000
lsof -i :5000
# Kill by PID
kill <PID>
```

On macOS, `ControlCe` (Control Center) also binds port 5000 sometimes — that conflict is fine; MLflow will still bind successfully.

### A3. Connecting your code to the server

In your Python code or notebook:

```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("pima_rf_gridsearch")
```

The experiment is created if it does not exist.

---

## Part B — Training and Comparing Models

We trained two model families and compared them:

1. RandomForest with a grid over `n_estimators` and `max_depth`
2. XGBoost with a grid over `n_estimators`, `max_depth`, `learning_rate`, `subsample`

### B1. Data loading

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "https://raw.githubusercontent.com/SubaashNair/dataset/refs/heads/main/datasets/medical/diabetes_clean.csv"
)

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

`stratify=y` keeps the class balance the same in train and test — important for medical classification where positives are rarer.

### B2. RandomForest grid search with MLflow logging

```python
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

mlflow.set_experiment("pima_rf_gridsearch")

n_estimators_list = [10, 20, 30, 40, 50]
max_depth_list = [3, 4, 5, None]

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        depth_label = max_depth if max_depth is not None else "inf"
        with mlflow.start_run(run_name=f"rf_n{n_estimators}_d{depth_label}"):
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": 42,
            }
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy":  accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall":    recall_score(y_test, y_pred),
                "f1":        f1_score(y_test, y_pred),
                "roc_auc":   roc_auc_score(y_test, y_proba),
            }

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("dataset", "pima_diabetes")

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=X_train[:5],
            )
```

**What each MLflow call does:**

| Call | Purpose |
|---|---|
| `mlflow.start_run(run_name=...)` | Creates a new run. Use a `with` block so it auto-closes. |
| `mlflow.log_params(...)` | Stores the hyperparameters. Visible in the UI. |
| `mlflow.log_metrics(...)` | Stores numeric outcomes. Can be plotted across runs. |
| `mlflow.set_tag(...)` | Free-form key/value. Useful for filtering in the UI. |
| `mlflow.sklearn.log_model(...)` | Serializes the trained model + saves a `MLmodel` spec file. |
| `infer_signature(...)` | Captures input/output schema. Lets MLflow validate inputs at serve time. |
| `input_example=...` | A few rows of sample input baked into the model artifact. |

**Why log a signature?** Without it, MLflow cannot validate inputs at serve time — your API will accept malformed JSON and crash inside the model. With a signature, you get a clean 400-style error.

### B3. XGBoost grid search

Same pattern, different model family:

```python
import mlflow.xgboost
import xgboost as xgb

param_grid = [
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,  "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,  "subsample": 1.0},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.01, "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,  "subsample": 0.7},
]

scale = (y_train == 0).sum() / (y_train == 1).sum()  # class-imbalance correction

for params in param_grid:
    run_name = f"xgb_n{params['n_estimators']}_d{params['max_depth']}_lr{params['learning_rate']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", "xgboost")
        mlflow.log_params(params)
        mlflow.log_param("scale_pos_weight", scale)

        model = xgb.XGBClassifier(
            **params,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        for name, fn in [
            ("accuracy", accuracy_score),
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score),
        ]:
            mlflow.log_metric(name, fn(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            input_example=X_test.iloc[:5],
        )
```

### B4. Comparing runs in the UI

Open `http://127.0.0.1:5000`.

1. Click your experiment name (`pima_rf_gridsearch`).
2. Tick the checkbox of the runs you want to compare.
3. Click **Compare**.
4. The metrics table appears side-by-side; the parallel-coordinates plot lets you see how hyperparameters drive metrics.

**What to look for:**

- Sort by `roc_auc` descending — best discrimination between positive and negative.
- Sort by `recall` — important for medical contexts where false negatives are costly.
- Filter by tag: `tags.model_type = 'xgboost'` to see only XGBoost runs.

In our case, XGBoost beat RandomForest on `roc_auc`, so the champion is XGBoost.

---

## Part C — Model Registry and Aliases

### C1. What the registry adds

The Tracking layer logs every experiment. The **Registry** layer is the curated list of models you actually want to deploy. It is a separate concept.

Each registered model has:

- A **name** (e.g. `pima_diabetes_rf_classifier`)
- One or more **versions** (`1`, `2`, `3`, …)
- Optional **aliases** that point to a specific version (`@champion`, `@challenger`, `@staging`)

You promote models by reassigning aliases, never by renaming versions. This is the key insight: **deployment is a pointer move**, not a copy.

### C2. Registering a model from the UI

1. In the MLflow UI, open the run with the best `roc_auc`.
2. Click the **model** artifact in the artifacts panel.
3. Click **Register Model**.
4. Type a name (e.g. `pima_diabetes_rf_classifier`).
5. Click **Register**.

This creates version 1 of the model.

### C3. Assigning the @champion alias

In the UI:

1. Go to **Models** in the top nav.
2. Click your registered model name.
3. Click the version you want to crown.
4. Add the alias `champion`.

In code:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name="pima_diabetes_rf_classifier",
    alias="champion",
    version="2",
)
```

Now you can reference the model in code as:

```
models:/pima_diabetes_rf_classifier@champion
```

Without ever changing the loading code, you can promote a new version by reassigning the alias — zero code change in your serving layer.

### C4. Loading a registered model

```python
import mlflow.xgboost

model = mlflow.xgboost.load_model("models:/pima_diabetes_rf_classifier@champion")
```

The URI format is `models:/<name>@<alias>` or `models:/<name>/<version>`.

---

## Part D — Validating the Model Before Serving

Before exposing anything to the network, sanity-check on a single sample:

```python
import pandas as pd
import mlflow.xgboost

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.xgboost.load_model("models:/pima_diabetes_rf_classifier@champion")

sample = pd.DataFrame([{
    "pregnancies": 2, "glucose": 138, "diastolic": 62,
    "triceps": 35, "insulin": 0, "bmi": 33.6,
    "dpf": 0.127, "age": 47,
}])

print(model.predict(sample))           # [1]
print(model.predict_proba(sample))     # [[0.35, 0.65]] roughly
```

If this fails, the deployment will fail too. Always validate in-process before going to the network.

---

## Part E — Serving the Model as a REST API

MLflow has a built-in scoring server. One command turns your registered model into an HTTP service:

```bash
mlflow models serve \
  -m "models:/pima_diabetes_rf_classifier@champion" \
  --port 5001 \
  --env-manager local
```

### E1. What each flag means

| Flag | What it does |
|---|---|
| `-m URI` | Which model to serve. Accepts a registry URI, run URI, or local path. |
| `--port 5001` | TCP port to bind. Default is 5000 but that collides with the tracking UI. |
| `--env-manager local` | Use your current Python env. Alternatives: `virtualenv`, `conda` (slower; recreates the env). |
| `--host` | Bind address. Default is `127.0.0.1`. Change to `0.0.0.0` inside containers (see Part G). |

### E2. Calling the API

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [{
      "pregnancies": 2, "glucose": 138, "diastolic": 62,
      "triceps": 35, "insulin": 0, "bmi": 33.6,
      "dpf": 0.127, "age": 47
    }]
  }'
```

Response:

```json
{"predictions": [1]}
```

### E3. Request formats MLflow accepts

| Format | Example |
|---|---|
| `dataframe_records` | `[{"col": val, ...}, ...]` — list of dicts (most readable) |
| `dataframe_split` | `{"columns": [...], "data": [[...], [...]]}` — compact |
| `instances` | TF-Serving compatible |

`dataframe_records` is the most ergonomic for teaching; the others are for tool interop.

### E4. The `/health` endpoint

```bash
curl http://127.0.0.1:5001/health
# 200 OK (empty body)
```

Use this in monitoring, in container healthchecks, and in your client UIs.

### E5. Default predictions, not probabilities

The scoring server exposes the model's `predict()` method only. For our binary classifier, that returns labels (0/1), not probabilities.

To expose `predict_proba`, you must wrap the model in a custom `mlflow.pyfunc.PythonModel`. See the MLflow docs on custom Python models — out of scope here.

---

## Part F — Containerizing with Docker (auto-generated)

MLflow can build a Docker image from a registered model in one command. The tracking server must be running when you build.

```bash
mlflow models build-docker \
  --model-uri "models:/pima_diabetes_rf_classifier@champion" \
  --name "pima-diabetes-api" \
  --env-manager local
```

### F1. Running the auto-generated image

```bash
# IMPORTANT: override the entrypoint to bind on 0.0.0.0,
# otherwise the server only listens on localhost INSIDE the container,
# and host port-forwards will not reach it.
docker run -d \
  --name pima-diabetes-server \
  -p 5001:8000 \
  --entrypoint mlflow \
  pima-diabetes-api \
  models serve \
    -m /opt/ml/model \
    --host 0.0.0.0 \
    --port 8000 \
    --env-manager local
```

Test:

```bash
sleep 15
curl -s http://127.0.0.1:5001/health
# 200
```

### F2. The localhost trap

Inside containers, `127.0.0.1` means "this container, nobody else." Docker's port forwarding (`-p host:container`) maps to the container's network interface — but if your app only listens on the loopback interface, no one outside can reach it.

**Always bind containerized servers to `0.0.0.0` (all interfaces).**

### F3. Why the image is 1.4 GB

The auto-generated image bundles a complete conda environment with every dependency the model needs. Conda is heavy. If you want a smaller image, hand-roll a Dockerfile — that is Part G.

### F4. Cleaning up

```bash
docker rm -f pima-diabetes-server          # Stop and remove the container
docker images | grep pima-diabetes-api      # See the image
docker rmi pima-diabetes-api                # Remove the image
```

---

## Part G — Custom Combined Dockerfile (Streamlit + MLflow API)

For deployment to HF Spaces we want a single container that runs both the MLflow scoring server and a Streamlit web UI. HF Spaces exposes one public port (`7860`); the MLflow API runs internally on `5001`.

### G1. Project layout

```
ML_FLOW_TEST/
├── Dockerfile
├── start.sh
├── requirements.txt
├── streamlit_client.py
├── model/                       # bundled champion artifact
│   ├── MLmodel
│   ├── model.ubj
│   ├── conda.yaml
│   ├── python_env.yaml
│   ├── requirements.txt
│   └── ...
└── README.md                    # with HF Spaces frontmatter
```

### G2. Bundling the model

You cannot rely on a tracking server at HF build time. Bake the model into the image:

```bash
# Copy the champion artifact directory out of mlruns
cp -r mlruns/3/models/m-<HASH>/artifacts ./model
ls ./model
# MLmodel  conda.yaml  input_example.json  model.ubj  ...
```

Find the right hash by:

```bash
find mlruns -name registered_model_meta
cat <found_path>
# model_name: pima_diabetes_rf_classifier
# model_version: '2'           ← matches the @champion alias
```

### G3. requirements.txt

Pin versions to match the saved model's environment (read `model/requirements.txt` to see what it expects):

```text
mlflow==3.12.0
xgboost==3.2.0
scikit-learn==1.8.0
pandas==2.3.3
numpy==2.4.3
pyarrow==23.0.1
psutil==7.2.2
scipy==1.17.1
streamlit>=1.30
requests>=2.31
```

**Why pin tightly?** sklearn serialized objects are version-sensitive. Loading a model trained on sklearn 1.8 inside an env with sklearn 1.3 gives cryptic `AttributeError`s or silent behavior changes. XGBoost UBJSON is more forgiving but still depends on schema compatibility.

### G4. Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ /opt/ml/model/
COPY streamlit_client.py start.sh ./
RUN chmod +x start.sh

ENV MODEL_API_URL=http://127.0.0.1:5001

EXPOSE 7860

CMD ["./start.sh"]
```

**Choices explained:**

- `python:3.12-slim` — matches the Python version recorded in `MLmodel`. Mismatching majors breaks deserialization.
- `curl` is the only system dep we need (for the health-check loop in `start.sh`).
- `pip install --no-cache-dir` keeps the image smaller; pip cache adds ~200 MB.
- `MODEL_API_URL` is an env var so the Streamlit app can be reconfigured at runtime without rebuilding.
- `EXPOSE 7860` is informational; the actual binding happens in `start.sh`.

### G5. start.sh

```bash
#!/usr/bin/env bash
set -e

mlflow models serve \
  -m /opt/ml/model \
  --host 0.0.0.0 \
  --port 5001 \
  --env-manager local &

echo "Waiting for MLflow scoring server on :5001..."
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:5001/health > /dev/null; then
    echo "MLflow scoring server is up."
    break
  fi
  sleep 1
done

exec streamlit run streamlit_client.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true
```

**Why `set -e`?** If `mlflow models serve` fails to launch, we want the container to exit, not silently run Streamlit alone.

**Why the loop?** MLflow's first start initializes a tracking DB and can take ~10–20 seconds. Streamlit needs the API ready before users hit it.

**Why `exec`?** Replaces the shell with Streamlit so signals (SIGTERM from Docker stop) reach Streamlit directly, allowing clean shutdown.

### G6. Building and running locally

```bash
docker build -t pima-demo:latest .

# Run with BOTH ports mapped so you can also hit the API directly
docker run --rm -p 7860:7860 -p 5001:5001 pima-demo:latest
```

Test:

```bash
# Streamlit UI
open http://localhost:7860

# Internal API also exposed for testing
curl -s -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"pregnancies":2,"glucose":138,"diastolic":62,"triceps":35,"insulin":0,"bmi":33.6,"dpf":0.127,"age":47}]}'
```

### G7. Image size note

Our slim image is about 2.5 GB. Most of it is `nvidia-nccl-cu12` (~600 MB) and `scipy/numpy/pandas` (~300 MB together). XGBoost pulls in CUDA libs even for CPU-only inference. To trim further you would need:

- A `xgboost-cpu` build (community wheels, less standard)
- Or installing xgboost with `--no-deps` and managing transitive deps manually

For a teaching demo, 2.5 GB is acceptable. HF Spaces accepts up to 50 GB.

---

## Part H — Streamlit Frontend

`streamlit_client.py` wraps the REST API in a clickable form.

### H1. Key design choices

| Choice | Reason |
|---|---|
| `requests` over `urllib` | More readable for students |
| `MODEL_API_URL` env var | Lets the same code run locally and in the container without code change |
| Two tabs (single + batch) | Tabs keep both modes equally discoverable; a radio buries one |
| `st.form` around inputs | Prevents an API call on every keystroke |
| Drop the probability bar | The default MLflow API returns labels only; faking a probability would mislead students |

### H2. Talking to the API

```python
import requests

def call_model_api(records: list[dict], api_url: str) -> list[int]:
    response = requests.post(
        f"{api_url}/invocations",
        json={"dataframe_records": records},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["predictions"]

def check_health(api_url: str) -> bool:
    try:
        return requests.get(f"{api_url}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False
```

### H3. Single prediction form

```python
import streamlit as st

with st.form("single_prediction"):
    cols = st.columns(2)
    inputs = {}
    for i, feature in enumerate(FEATURE_COLUMNS):
        with cols[i % 2]:
            inputs[feature] = st.number_input(
                feature, value=DEFAULTS[feature], help=HELP[feature], format="%.3f",
            )
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        predictions = call_model_api([inputs], api_url)
        st.success("Diabetic" if predictions[0] == 1 else "Not Diabetic")
    except requests.RequestException as e:
        st.error(f"API error: {e}")
```

### H4. Running it locally

```bash
streamlit run streamlit_client.py
```

Set the env var to point at any API URL:

```bash
MODEL_API_URL=http://example.com:5001 streamlit run streamlit_client.py
```

---

## Part I — Pushing to GitHub

### I1. Initialize a fresh repo

If your project lives inside another git repo (like ours did inside the larger SPD repo), `git init` inside the subfolder creates a nested independent repo. That is fine — commands run from inside the subdirectory use the inner `.git`.

```bash
cd ML_FLOW_TEST
git init -b main
```

### I2. .gitignore (essential)

```text
mlruns/
mlflow.db
.idea/
.ruff_cache/
__pycache__/
*.pyc
.venv/
venv/
.env
.DS_Store
```

**Why exclude `mlruns/` and `mlflow.db`?**

- `mlruns/` can grow to gigabytes; it is your local artifact store.
- `mlflow.db` is your local SQLite tracking DB; not portable between machines.
- They are reproducible from the notebook anyway.

### I3. .dockerignore

```text
.git/
.github/
mlruns/
mlflow.db
.idea/
.ruff_cache/
__pycache__/
*.pyc
.venv/
venv/
.env
.DS_Store
mlflow.ipynb
docker.md
main.py
```

Same idea as `.gitignore` but applied at `docker build` time. Excluding the notebook and docs keeps the build context small and the image lean.

### I4. First commit

```bash
git add .
git status                    # verify mlruns/ and mlflow.db are NOT staged
git commit -m "feat: initial MLflow Pima diabetes demo"
```

### I5. Creating the GitHub repo

If you have `gh` authenticated:

```bash
gh repo create pima-diabetes-mlflow \
  --public \
  --source=. \
  --remote=origin \
  --description "End-to-end MLflow demo: train, register, serve, deploy"
```

This creates the repo on GitHub *and* adds it as the `origin` remote in one shot.

Then push:

```bash
git push -u origin main
```

The `-u` flag sets `origin/main` as the upstream branch, so future `git push` and `git pull` don't need arguments.

---

## Part J — Deploying to Hugging Face Spaces

HF Spaces are git repos hosted by Hugging Face. With `sdk: docker` the Space rebuilds and runs your `Dockerfile` on every push.

### J1. README.md frontmatter (required)

HF reads YAML frontmatter at the top of `README.md` to configure the Space:

```markdown
---
title: Pima Diabetes MLflow Demo
emoji: 🩺
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---
```

**Fields:**

| Field | Purpose |
|---|---|
| `title` | Shown in the Space card |
| `emoji`, `colorFrom`, `colorTo` | Visual styling for the card |
| `sdk` | `docker` here; otherwise `streamlit`/`gradio`/`static` |
| `app_port` | Which port your container listens on for HTTP. **Must match `start.sh`** |
| `pinned` | Show in your profile's pinned section |

### J2. Authenticate the HF CLI

Get a write-scope token from https://huggingface.co/settings/tokens

```bash
huggingface-cli login
# Paste token when prompted
```

Verify:

```bash
hf auth whoami
# user: <your-username>
```

### J3. Create the Space

```bash
hf repo create pima-diabetes-mlflow \
  --repo-type space \
  --space_sdk docker
```

This creates the empty Space at `https://huggingface.co/spaces/<your-user>/pima-diabetes-mlflow`.

### J4. Add the Space as a remote and push

```bash
git remote add hf https://huggingface.co/spaces/<your-user>/pima-diabetes-mlflow
git remote -v
# origin   git@github.com:You/pima-diabetes-mlflow.git
# hf       https://huggingface.co/spaces/<your-user>/pima-diabetes-mlflow
```

The freshly created Space has an auto-generated `README.md` and `.gitattributes`. Merge those into your branch first:

```bash
git pull hf main --allow-unrelated-histories --no-rebase --no-edit -X ours
```

`-X ours` resolves conflicts in our favor (we want our `README.md`, not HF's default).

### J5. Handling binary files: Git LFS

HF Spaces **rejects** any binary file not tracked by Git LFS, regardless of size. If your repo contains a `.ubj`, `.h5`, etc., you must tell Git to use LFS for those extensions.

The default `.gitattributes` HF ships includes many common ML formats but **not** XGBoost UBJSON. Add it:

```bash
# View current rules
cat .gitattributes | head -5

# Add the UBJ rule
echo "*.ubj filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
git add .gitattributes
```

**Important:** if you already committed binary files before adding the LFS rule, you must rewrite history to convert them to LFS pointers:

```bash
git lfs install
git lfs migrate import --include="*.ubj" --everything

# Verify
git lfs ls-files
# 57b092dca2 - model/model.ubj   ← now stored as LFS pointer
```

`git lfs migrate import` rewrites every commit that touched the file. The file content moves to LFS storage; the commits now contain only pointer files.

### J6. Push to the Space

If you used `git lfs migrate`, history was rewritten — you now need a force push:

```bash
git push hf main --force
# Uploading LFS objects: 100% (1/1), 458 KB | 0 B/s, done.
# + 8a3cf8e...0a46237 main -> main (forced update)
```

If you did the LFS setup before the first commit, a plain `git push hf main` is enough.

### J7. Watching the build

Go to your Space URL → click **Logs**. The first build downloads the base image, installs every pip dep, and copies the model. Expect 5–8 minutes cold.

When the build is green, the **App** tab shows your Streamlit UI live at `https://huggingface.co/spaces/<your-user>/pima-diabetes-mlflow`.

### J8. Why HF rebuilds the image (not pull yours)

HF runs on x86_64 amd64. Your Mac probably built an arm64 image. Pushing source (Dockerfile + code) and letting HF rebuild on its own infrastructure neatly side-steps multi-arch builds. The local build is just for verification.

---

## Part K — CI/CD: GitHub Actions Sync to HF

The manual flow is: edit → commit → `git push origin` → `git push hf`. Tedious. A GitHub Action removes the second push.

### K1. The workflow file

Create `.github/workflows/sync-to-hf.yml`:

```yaml
name: Sync to Hugging Face Space

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout (with LFS)
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true

      - name: Push to HF Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USER: <your-hf-username>
          HF_SPACE: pima-diabetes-mlflow
        run: |
          git push --force "https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}" main
```

**Why each line matters:**

| Line | Purpose |
|---|---|
| `on.push.branches: [main]` | Run on every push to main |
| `on.workflow_dispatch` | Also allow manual triggers from the Actions tab |
| `fetch-depth: 0` | Get full history; HF needs to see the same commits |
| `lfs: true` | Pull actual LFS content. Without this, only pointer files are checked out. |
| `--force` | Required if local history was ever rewritten; safe for a downstream mirror |

### K2. Adding the HF_TOKEN secret

The action uses a secret named `HF_TOKEN`. **Never commit the token to the repo.**

Add it via the GitHub CLI:

```bash
gh secret set HF_TOKEN -R <user>/pima-diabetes-mlflow
# Paste your HF write-scope token when prompted
```

Or via the web UI: Settings → Secrets and variables → Actions → New repository secret. Name = `HF_TOKEN`, value = your HF token.

### K3. Verifying the action

```bash
git push origin main
gh run list -R <user>/pima-diabetes-mlflow --limit 3
# completed  success  Sync to Hugging Face Space  main  push  ...
```

Inspect the latest run:

```bash
gh run view <run-id> -R <user>/pima-diabetes-mlflow --log | tail -30
```

You should see:

```
Push to HF Space    HF_TOKEN: ***
                    To https://huggingface.co/spaces/<user>/pima-diabetes-mlflow
                       <old>..<new>  main -> main
```

### K4. From here on

Workflow becomes:

```bash
git add .
git commit -m "feat: ..."
git push origin main
# Action mirrors to HF automatically
# Space rebuilds within seconds of the push
```

---

## Part L — Why this app won't run on Streamlit Community Cloud as-is

A common question: "I already have a Streamlit app — why can't I just deploy it to share.streamlit.io and skip everything else?" Short answer: because the app is **two-tier**, and Streamlit Community Cloud only hosts the UI tier.

### L1. The architecture mismatch

What the current app does at every "Predict" click:

```
[Browser] ──► [Streamlit process] ──HTTP──► [MLflow scoring server] ──► [model.predict()]
                                            (at http://127.0.0.1:5001)
```

There are **two separate processes** — Streamlit on one port, the MLflow scoring server on another. Inside our Docker container they coexist, so `127.0.0.1:5001` resolves to the MLflow process running next to Streamlit. That works because both processes are inside the same network namespace.

Streamlit Community Cloud runs **only your Streamlit process**. There is no second container, no MLflow server next door. When Streamlit Cloud's Python process tries to `requests.post("http://127.0.0.1:5001/invocations", ...)`, it hits nothing — `127.0.0.1` on Streamlit Cloud's runner is Streamlit Cloud, and Streamlit Cloud has no MLflow server.

This is not a bug, a config gap, or a missing token. It is a fundamental architecture mismatch.

### L2. The three ways to bridge the gap

| Option | What you change | Cost | Trade-off |
|---|---|---|---|
| **A. Embed the model** | Load `model/` directly in Python, drop the HTTP call | Free | Loses the REST-API teaching surface |
| **B. Deploy the API publicly** | Host MLflow on a second service, point Streamlit at its URL | $ on the API tier | Keeps two-tier teaching architecture |
| **C. Stay on HF Spaces** | Do nothing | Free | UI domain is `huggingface.co/spaces/...` not `*.streamlit.app` |

### L3. Option A — Embed the model in Streamlit (recommended)

Streamlit Community Cloud is built for single-process apps. Collapse the two tiers into one by loading the model directly.

**Code change.** Add at the top of `streamlit_client.py`:

```python
import mlflow.xgboost
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model():
    """Load once per Streamlit session; reused across reruns."""
    return mlflow.xgboost.load_model("model")  # relative path to the bundled artifact

model = load_model()
```

Then replace HTTP calls. Where you had:

```python
predictions = call_model_api(records, api_url)
```

write:

```python
predictions = model.predict(pd.DataFrame(records)).tolist()
```

Delete `check_health` and the sidebar URL input — there is no remote API anymore. The "curl equivalent" expander becomes irrelevant; replace it with a "Python equivalent" if you want to keep teaching the underlying call.

**`@st.cache_resource` is important.** Without it, Streamlit reloads the model on every interaction (every form submission, every rerun). With it, the model is loaded once per session and kept in memory.

**Two extra files you need:**

1. **`requirements.txt`** — Streamlit Cloud reads this. The existing one already pins `mlflow`, `xgboost`, `scikit-learn`, etc.
2. **`packages.txt`** — system-level apt packages. Add one line:

```text
git-lfs
```

This tells Streamlit Cloud to install `git-lfs` so it can fetch the actual `model.ubj` content when cloning your repo. Without it, the runner gets the LFS pointer file (~100 bytes saying "this is an LFS file"), and `load_model("model")` crashes because `model.ubj` is unreadable.

**Connecting the repo.**

1. Go to https://share.streamlit.io
2. Click **New app** → pick your `pima-diabetes-mlflow` repo
3. Main file path: `streamlit_client.py`
4. Branch: `main`
5. **Advanced settings** → leave Python version at default (3.12+) or pin to match
6. Click **Deploy**

First boot takes ~3–5 minutes (LFS fetch + pip install). Subsequent deploys re-run on every `git push origin main`.

### L4. Option B — Deploy the API separately

Keep the two-tier architecture by giving the MLflow scoring server a public URL.

**Picking a host.** Any of these work; pick by your familiarity:

| Host | Free tier | Cold start | Notes |
|---|---|---|---|
| Fly.io | Generous, Docker-first | ~5s | Best DX for Docker; `fly launch` from your Dockerfile |
| Railway | $5/mo credit | ~10s | Easiest UI; one-click Docker deploys |
| Render | Free for low traffic | ~30s on cold | Sleeps after 15 min inactivity on free tier |
| Google Cloud Run | Pay-per-use, scales to zero | ~1–3s | Cheapest at low volume; needs `gcloud` setup |

**Modify the Dockerfile.** For an API-only deployment, strip the Streamlit half from `start.sh`:

```bash
#!/usr/bin/env bash
set -e
exec mlflow models serve \
  -m /opt/ml/model \
  --host 0.0.0.0 \
  --port ${PORT:-5001} \
  --env-manager local
```

Most platforms inject a `$PORT` env var the app must bind to. The `${PORT:-5001}` default makes the same image work locally and on the platform.

**Wire Streamlit Cloud to the API.**

In Streamlit Cloud's app settings → **Secrets**, add:

```toml
MODEL_API_URL = "https://your-mlflow-api.fly.dev"
```

Your existing code already reads this env var, so no code change in `streamlit_client.py`. Add `os.environ` shim if you want to also read from `st.secrets`:

```python
import os
import streamlit as st

DEFAULT_API_URL = (
    st.secrets.get("MODEL_API_URL")
    or os.environ.get("MODEL_API_URL")
    or "http://127.0.0.1:5001"
)
```

### L5. Option C — Just use the HF Space

The HF Space at `https://huggingface.co/spaces/<user>/<space>` already publicly serves the Streamlit UI with the model embedded inside the same container. That is effectively Option A, already deployed, already public, already free. No additional work.

The only reason to prefer Streamlit Community Cloud is the `*.streamlit.app` URL or its integration with Streamlit's own analytics dashboard. Otherwise the HF Space gives the same experience to your students.

### L6. Decision matrix

| If you want… | Pick |
|---|---|
| Free, fastest path, on `*.streamlit.app` | Option A (embed model) |
| To teach the REST API tier explicitly | Option B (separate API) |
| Already done, no extra work | Option C (HF Space) |

### L7. Common mistakes when deploying to Streamlit Community Cloud

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: model.ubj` on boot | LFS not enabled on the runner | Add `packages.txt` with `git-lfs` |
| `ModuleNotFoundError: No module named 'mlflow'` | Streamlit Cloud installed only what it heuristically detected | Ensure `requirements.txt` is at repo root, not in a subfolder |
| App boots but "API unreachable" in sidebar | Tried Option A but left the `requests` code in | Remove `call_model_api` and call `model.predict()` directly |
| Build hangs on `pip install xgboost` | Free tier sometimes hits memory limits installing big wheels | Use `xgboost==3.2.0` (pre-built CPU wheels) and remove `nvidia-*` from deps |

---

## Troubleshooting

### Build / runtime issues

**MLflow scoring server on `127.0.0.1` not reachable from outside the container.**

Symptom: `curl ... /health` returns no response, container logs show server running.

Fix: bind on `0.0.0.0` instead of `127.0.0.1`. See Part G5.

---

**Sklearn `AttributeError` or deserialization errors at serve time.**

Symptom: errors like `Can't get attribute '_BaseHeterogeneousEnsemble' on <module 'sklearn.ensemble._base'>` or similar.

Cause: version mismatch between training env and serving env.

Fix: pin `requirements.txt` to the exact versions in `model/requirements.txt`. See Part G3.

---

**Python version mismatch.**

Symptom: `unsupported protocol: 5` or weird import errors.

Cause: model saved with Python 3.12, served with 3.11 (or vice versa).

Fix: check `MLmodel` → `python_version`, match it in your Dockerfile `FROM`.

---

**Platform mismatch warning when running Docker locally.**

Symptom: `WARNING: The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8)...`

Cause: image built for a different CPU architecture than your host.

Fix: ignore for local testing if it works (Rosetta on Apple Silicon handles this). For deploys, let HF Spaces rebuild on its own infra rather than pushing an image.

---

### GitHub / HF push issues

**HF rejects push: "contains binary files."**

Symptom:
```
remote: Your push was rejected because it contains binary files.
remote: Offending files:
remote:   - model/model.ubj
```

Fix: add the file extension to `.gitattributes` with LFS filter and run `git lfs migrate import`. See Part J5.

---

**HF rejects push: "non-fast-forward."**

Symptom: `! [rejected] main -> main (non-fast-forward)`

Cause: local history was rewritten (e.g. by `git lfs migrate`) so the local branch is not a descendant of the remote.

Fix on a fresh Space: `git push hf main --force`. Safe because the Space had no real prior content.

Fix on a Space with history you care about: `git pull hf main --rebase` and resolve.

---

**GitHub Action fails with `Authentication failed`.**

Cause: `HF_TOKEN` secret missing or expired.

Fix: rerun `gh secret set HF_TOKEN -R <repo>` with a fresh write-scope token.

---

### Local dev issues

**Port 5000 already in use.**

Symptom: `Address already in use` when running `mlflow ui`.

Diagnose:

```bash
lsof -i :5000
```

Most common culprits on macOS:
- macOS Control Center (`ControlCe`) — harmless, MLflow will bind anyway
- A previous `mlflow ui` session you forgot to stop — `kill <PID>`

---

**Streamlit shows "API unreachable" in the sidebar.**

Diagnose:

```bash
# Is the model server up?
curl http://127.0.0.1:5001/health

# Is the container running?
docker ps --filter name=pima-diabetes-server
```

Fix: start the container per Part F1 or run `mlflow models serve` directly.

---

## Glossary

| Term | Meaning |
|---|---|
| **Run** | One execution of training. Has params, metrics, tags, artifacts. |
| **Experiment** | A named container for related runs. |
| **Artifact** | Any file produced by a run — usually the serialized model. |
| **Signature** | Schema describing model inputs and outputs. Used for validation at serve time. |
| **Registry** | The curated list of models you want to deploy. Separate from Tracking. |
| **Alias** | A movable pointer to a model version. `@champion`, `@staging`, etc. |
| **`models:/` URI** | The MLflow scheme for referencing registered models. |
| **`/invocations`** | The MLflow scoring server's prediction endpoint. |
| **`pyfunc`** | MLflow's universal model interface — any model can be wrapped as a `PythonModel`. |
| **HF Space** | A hosted git repo on Hugging Face that runs your app. |
| **Docker SDK Space** | A Space that runs your `Dockerfile` directly, instead of using a built-in runtime. |
| **LFS** | Git Large File Storage. Stores binaries outside git history; commits contain pointers. |
| **Xet** | Hugging Face's evolution of LFS for very large model files. |

---

## Quick command reference

```bash
# Start tracking server (leave running in its own terminal)
mlflow ui

# Build and run combined container locally
docker build -t pima-demo .
docker run --rm -p 7860:7860 -p 5001:5001 pima-demo

# Create GitHub repo + push
gh repo create pima-diabetes-mlflow --public --source=. --remote=origin
git push -u origin main

# Create HF Space + push (one-time)
huggingface-cli login
hf repo create pima-diabetes-mlflow --repo-type space --space_sdk docker
git remote add hf https://huggingface.co/spaces/<user>/pima-diabetes-mlflow
git pull hf main --allow-unrelated-histories --no-rebase --no-edit -X ours
git push hf main

# Migrate a binary file into LFS retroactively
git lfs install
git lfs migrate import --include="*.ubj" --everything
git push hf main --force

# Set up GitHub Action sync
gh secret set HF_TOKEN -R <user>/pima-diabetes-mlflow
# Commit .github/workflows/sync-to-hf.yml and push

# Daily workflow after setup
git add .
git commit -m "feat: ..."
git push origin main
# Action mirrors to HF, Space rebuilds
```
