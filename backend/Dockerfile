# Build context is the repository root (see docker/docker-compose.yml), because the
# backend installs the sibling `ai-services` package via `-e ../ai-services`.
FROM python:3.11-slim

# libgl1 / libglib2.0-0 are required by opencv-python-headless at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY ai-services ./ai-services
COPY backend ./backend

RUN cd backend \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace/backend

ENV VISIONIQ_VECTOR_STORE_DIR=/data/vector_store \
    VISIONIQ_DATABASE_URL=sqlite:////data/visioniq.db \
    VISIONIQ_MODEL_CACHE=/data/model_cache

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
