# ============================================================
# JARVIS — Production Container Image
# APEX LEGENDS | Nebius × NVIDIA Global AI Hackathon 2026
# ============================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    JARVIS_ENV=production \
    JARVIS_DATA_DIR=/app/data \
    JARVIS_HOST=0.0.0.0

WORKDIR /app

# Install security updates and curl for container health checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY pyproject.toml README.md ./
COPY jarvis/ ./jarvis/

# Create persistent storage mount directory and non-root execution user
RUN mkdir -p /app/data && \
    useradd -u 10001 -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to least-privilege non-root user
USER appuser

# Declare persistent data volume mount point
VOLUME ["/app/data"]

# Expose default HTTP port
EXPOSE 8000

# Container liveness health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-8000}/healthz || exit 1

# Production startup command honoring dynamic cloud $PORT
CMD ["sh", "-c", "python -m jarvis.main serve --host 0.0.0.0 --port ${PORT:-8000}"]
