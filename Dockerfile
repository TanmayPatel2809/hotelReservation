# --- STAGE 1: The Builder (Temporary and Secure) ---
FROM python:slim AS builder

# Accept credentials from the Jenkins build-arg. This is temporary.
ARG GOOGLE_APPLICATION_CREDENTIALS_CONTENT

# Create a credentials file that will ONLY exist in this builder stage
RUN echo "${GOOGLE_APPLICATION_CREDENTIALS_CONTENT}" > /tmp/gcp-credentials.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials.json

WORKDIR /app

# Copy source code and install dependencies
COPY . .
RUN pip install --no-cache-dir -e .

# Run training using the credentials. The resulting models are saved in this stage.
RUN python pipeline/training_pipeline.py


# --- STAGE 2: The Final Image (Clean and Secure) ---
FROM python:slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/

# Copy your application code AND the trained models from the builder.
# The temporary credentials file from /tmp is NOT copied over.
COPY --from=builder /app /app

EXPOSE 5000

# Run the application with the pre-trained models.
CMD ["python", "application.py"]