# syntax=docker/dockerfile:1
FROM python:slim AS builder

WORKDIR /app

# Copy source code and install dependencies
COPY . .
RUN pip install --no-cache-dir -e .

# Mount the secret and run training
RUN --mount=type=secret,id=gcp-key,target=/tmp/gcp-credentials.json \
    GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials.json \
    python pipeline/training_pipeline.py

# --- STAGE 2: The Final Image (Clean and Secure) ---
FROM python:slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/

# Copy your application code AND the trained models from the builder
COPY --from=builder /app /app

EXPOSE 5000

CMD ["python", "application.py"]