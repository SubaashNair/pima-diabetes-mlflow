# Dockerizing the Pima Diabetes Model (Mac & Windows)

End-to-end guide to packaging the MLflow-trained Pima Diabetes classifier as a Docker image, running it as a REST API, and calling it from any client.

This guide assumes you have already:
- Trained models with the notebook `mlflow.ipynb`
- Registered a model in the MLflow Model Registry under the name `pima_diabetes_rf_classifier`
- Assigned the alias `@champion` to the version you want to deploy

---

## Quick reference (the 4 commands)

For students who want the TL;DR. Detailed explanations follow below.

```bash
# 1. Build the image (run from the project root, MLflow UI must be running on :5000)
mlflow models build-docker \
  --model-uri "models:/pima_diabetes_rf_classifier@champion" \
  --name "pima-diabetes-api" \
  --env-manager local

# 2. Run the container (detached, with bind-host override)
docker run -d \
  --name pima-diabetes-server \
  -p 5001:8000 \
  --entrypoint mlflow \
  pima-diabetes-api \
  models serve -m /opt/ml/model --host 0.0.0.0 --port 8000 --env-manager local

# 3. Verify it's running
docker ps --filter name=pima-diabetes-server

# 4. Test the endpoint
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"pregnancies":2,"glucose":138,"diastolic":62,"triceps":35,"insulin":0,"bmi":33.6,"dpf":0.127,"age":47}]}'
```

Expected response:
```json
{"predictions": [1]}
```

---

## Prerequisites

### Mac (Apple Silicon M1/M2/M3/M4 or Intel)

1. **Install Docker Desktop**
   - Download: <https://www.docker.com/products/docker-desktop/>
   - Pick the build matching your chip: `Apple Silicon` for M-series, `Intel chip` for older Macs.
   - Open the app once after install — it provisions a Linux VM and a Docker daemon.
2. **Verify**:
   ```bash
   docker --version    # CLI version
   docker info         # Talks to the daemon — must succeed
   ```
   If `docker info` errors with "Cannot connect to the Docker daemon," Docker Desktop is not running. Open it from Applications.

### Windows

1. **Install Docker Desktop for Windows**
   - Download: <https://www.docker.com/products/docker-desktop/>
   - During install, accept the option to enable **WSL 2 backend** (required on Windows 10/11).
2. **Verify** in PowerShell or Command Prompt:
   ```powershell
   docker --version
   docker info
   ```
3. **Shell choice**: every `bash`-style command in this guide works in:
   - **Git Bash** (recommended — handles backslash line-continuations cleanly)
   - **WSL 2 Ubuntu** (most Linux-like experience)
   - **PowerShell** — but you must replace `\` line continuations with backticks (`` ` ``)

### Both platforms

You also need:
- A running **MLflow Tracking Server** on `http://127.0.0.1:5000`
  ```bash
  mlflow ui
  ```
- A **registered model** under `models:/pima_diabetes_rf_classifier@champion`

---

## Step 1: Build the Docker image

```bash
mlflow models build-docker \
  --model-uri "models:/pima_diabetes_rf_classifier@champion" \
  --name "pima-diabetes-api" \
  --env-manager local
```

What each flag does:

| Flag | Purpose |
|---|---|
| `--model-uri` | Where MLflow finds the model. `models:/<NAME>@<ALIAS>` resolves through the registry. |
| `--name` | The Docker image name you'll reference later. |
| `--env-manager local` | Uses pip (no conda) to install dependencies inside the image. Faster build, smaller image. |

Build takes **2–10 minutes** the first time (pulling Python base image + installing scikit-learn/xgboost/MLflow). Subsequent rebuilds are much faster thanks to Docker layer caching.

Verify the image exists:
```bash
docker images | grep pima-diabetes-api
```

You should see a row with `~1.4 GB` size.

> **Apple Silicon note**: MLflow currently produces `linux/amd64` images by default. On M-series Macs this means containers run under Rosetta/QEMU emulation — functional, but **2–5× slower** than native. For production, rebuild the Dockerfile MLflow generates with `docker buildx build --platform linux/arm64`. For teaching purposes, the emulated version is fine.

---

## Step 2: Run the container

This is the step where most students hit issues, so we use the **robust form**:

```bash
docker run -d \
  --name pima-diabetes-server \
  -p 5001:8000 \
  --entrypoint mlflow \
  pima-diabetes-api \
  models serve -m /opt/ml/model --host 0.0.0.0 --port 8000 --env-manager local
```

Why each piece matters:

| Piece | Purpose |
|---|---|
| `-d` | **Detached** mode — container runs in the background. Your terminal/notebook cell returns immediately. |
| `--name pima-diabetes-server` | Fixed container name. Makes cleanup (`docker rm -f <name>`) reliable. |
| `-p 5001:8000` | Maps **host port 5001** → **container port 8000**. Use `5001` because macOS reserves port 5000 for AirPlay (see Troubleshooting). |
| `--entrypoint mlflow` | Overrides the default entrypoint. The MLflow-generated entrypoint binds to `127.0.0.1` inside the container, which is unreachable from outside. We replace it. |
| `models serve -m /opt/ml/model --host 0.0.0.0` | **Forces `0.0.0.0`** so the server accepts connections from Docker's bridge network. This is the critical fix. |

Verify the container is running:
```bash
docker ps --filter name=pima-diabetes-server
```

You should see a row with `STATUS = Up X seconds`.

To watch the logs (Ctrl+C to exit, won't stop the container):
```bash
docker logs -f pima-diabetes-server
```

Wait until you see `Application startup complete.` lines from gunicorn workers — that means the model is loaded and the API is ready.

---

## Step 3: Call the API

### Health check

```bash
curl http://127.0.0.1:5001/health
```

Expected: `OK` (or HTTP 200 with no body).

### Real prediction

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {"pregnancies":2,"glucose":138,"diastolic":62,"triceps":35,
       "insulin":0,"bmi":33.6,"dpf":0.127,"age":47}
    ]
  }'
```

Expected response:
```json
{"predictions": [1]}
```

A `1` means the model predicts diabetes; `0` means non-diabetic.

### Python client example

```python
import requests

response = requests.post(
    "http://127.0.0.1:5001/invocations",
    json={
        "dataframe_records": [{
            "pregnancies": 2, "glucose": 138, "diastolic": 62,
            "triceps": 35, "insulin": 0, "bmi": 33.6,
            "dpf": 0.127, "age": 47
        }]
    },
)
print(response.json())
```

---

## Step 4: Stop & clean up

```bash
# Stop and remove the running container
docker rm -f pima-diabetes-server

# (Optional) remove the image too
docker rmi pima-diabetes-api
```

---

## Troubleshooting

### "Cannot connect to the Docker daemon" / "FileNotFoundError: docker.sock"

Docker Desktop isn't running. Open it from Applications (Mac) or the Start Menu (Windows). Wait for the whale icon in the system tray to stop animating, then retry.

### Port 5000 already in use (macOS)

macOS reserves port 5000 for **AirPlay Receiver**. Either:
- Use a different port: `mlflow ui --port 5050` and `docker run -p 5050:8000 ...`
- Or disable AirPlay Receiver: **System Settings → General → AirDrop & Handoff → AirPlay Receiver: Off**

### `curl: (56) Recv failure: Connection reset by peer`

The container is running but the app inside is bound to `127.0.0.1` (loopback), unreachable from outside the container. This is exactly what the `--entrypoint mlflow ... --host 0.0.0.0` override in Step 2 fixes. Make sure you're using the robust form, not the bare `docker run -p 5001:8000 pima-diabetes-api`.

### `WARNING: requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8)`

Apple Silicon Mac running an amd64 image — emulation mode is on. Container will work, just slowly. To go native, rebuild for `linux/arm64` (see "Apple Silicon note" above).

### `Error response from daemon: pull access denied for pima-diabetes-api`

Misleading error — it usually means you passed `--platform linux/arm64` to `docker run` but the local image is amd64-only. Docker tried to pull an arm64 variant and failed. **Drop the `--platform` flag**; use whatever platform you built for.

### `docker stop $(docker ps -q --filter ancestor=...)` errors with "requires at least 1 argument"

The filter found no running container, so `$()` expanded to empty. Use the safer pattern:
```bash
docker rm -f pima-diabetes-server 2>/dev/null || true
```

### Container starts but `/health` hangs for 30+ seconds

Under amd64 emulation on Apple Silicon, first-time startup is slow because:
1. The Python interpreter starts under Rosetta
2. Gunicorn forks 4 workers, each loads the model
3. The MLflow scoring server initializes an embedded SQLite DB

Just wait. Subsequent requests are fast.

### Lots of `sqlite3.OperationalError: table experiments already exists` warnings

Harmless. Multiple gunicorn workers race to initialize the same SQLite tracking DB; the first one wins, the rest log the warning. Ignore unless predictions actually fail.

### Windows-specific: backslash line continuations break in PowerShell

PowerShell uses backtick (`` ` ``) for line continuation, not backslash. Either:
- Convert all `\` to `` ` `` in commands
- Or run the commands as single-line versions
- Or use Git Bash / WSL 2 instead

### Windows-specific: WSL 2 doesn't see Docker

Open Docker Desktop → **Settings → Resources → WSL Integration** → enable the toggle for your distro (e.g. Ubuntu).

---

## What gets built into the image?

For curiosity / exam questions:

```bash
docker history pima-diabetes-api      # see all the build layers
docker inspect pima-diabetes-api      # full image metadata
```

Inside the image:
- `/opt/ml/model/` — your serialized MLflow model artifact (model file + signature + conda.yaml)
- Python 3.x + pip dependencies from your model's `requirements.txt`
- MLflow's scoring server (gunicorn + uvicorn + FastAPI) listening on port 8000

The image is ~1.4 GB because it bundles a full Python runtime + scikit-learn + xgboost + MLflow. A small serialized model becomes a 1.4 GB image — that's the cost of "ship everything needed to run it."

---

## Next steps

- **Deploy to a cloud registry**: `docker tag pima-diabetes-api <user>/pima-diabetes-api && docker push <user>/pima-diabetes-api`
- **Run on a managed platform**: AWS ECS, Google Cloud Run, Azure Container Apps — all accept a Docker image directly.
- **Add a load balancer + auth + HTTPS** for production. The MLflow scoring server is *not* production-ready as-is — wrap it behind a proper API gateway.
