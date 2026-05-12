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
