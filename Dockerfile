FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Node is only needed for stdio MCP servers launched via `npx` (e.g. mcp-remote for Zoho).
# Remove this line if your agent only uses HTTP MCP servers - smaller image, smaller attack surface.
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY agent ./agent

# Never run as root
RUN useradd --create-home --uid 10001 agent && mkdir -p /app/logs /app/data && chown agent /app/logs /app/data
USER agent

# Hosting platforms (Render, Railway, Cloud Run...) tell the app which port to use via $PORT.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=90s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')" || exit 1

# 1 worker: sessions + rate limits live in memory. Scale with more containers + Redis (see README).
# Shell form so ${PORT} is expanded; exec so uvicorn receives SIGTERM directly (clean shutdown).
CMD exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'
