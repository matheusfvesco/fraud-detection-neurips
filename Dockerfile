FROM ghcr.io/astral-sh/uv:0.8.2-python3.13-bookworm-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential openjdk-17-jdk && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr

COPY pyproject.toml uv.lock ./
COPY fraud_detection/ ./fraud_detection/
COPY models/ ./models/
COPY README.md LICENSE ./

RUN uv sync

CMD ["uv", "run", "fastapi", "run", "fraud_detection/api.py"]