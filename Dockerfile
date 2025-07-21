FROM python:slim AS builder

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -e .

RUN --mount=type=secret,id=gcp-key,target=/tmp/gcp-credentials.json \
    GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials.json \
    python pipeline/training_pipeline.py

FROM python:slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/

COPY --from=builder /app /app

EXPOSE 8080

CMD ["python", "application.py"]