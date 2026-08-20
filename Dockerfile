FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/storage/models

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY evaluation ./evaluation
COPY web ./web

RUN chmod +x scripts/docker_entrypoint.sh \
    && mkdir -p storage

EXPOSE 8000

ENTRYPOINT ["./scripts/docker_entrypoint.sh"]
