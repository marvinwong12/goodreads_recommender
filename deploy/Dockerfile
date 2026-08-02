# Build context is deploy/ itself (see prepare_deploy.sh) -- it only
# contains the handful of files the serving app actually needs, not the
# ~16GB training repo this was staged from.
FROM python:3.13-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY app/ app/
COPY data/ data/
COPY models/ models/

# Cloud Run injects the PORT env var (default 8080) and expects the app to
# listen on it; shell-form CMD is required so ${PORT} actually expands.
ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
