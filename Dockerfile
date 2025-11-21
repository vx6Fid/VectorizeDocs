FROM python:3.11-slim

WORKDIR /app

# Ensure transformers / ST cache is stored in a persistent path inside image
ENV HF_HOME=/app/.cache

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SentenceTransformer model (cached under HF_HOME)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY . .

EXPOSE 8000

# Gunicorn config
ENV GUNICORN_WORKERS=4
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000

CMD ["sh", "-c", "gunicorn app.main:app \
    -w ${GUNICORN_WORKERS} \
    -k uvicorn.workers.UvicornWorker \
    --bind ${APP_HOST}:${APP_PORT} \
    --preload \
    --timeout 120 \
    --graceful-timeout 120"]
