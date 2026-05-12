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
