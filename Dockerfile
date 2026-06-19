FROM python:3.12-slim

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (CPU-only, cached layer) ────────────────────
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# ── Application code (inference only) ───────────────────────
COPY src/ src/
COPY app/ app/
COPY .streamlit/ .streamlit/

# Create directories for model checkpoints and sample images
RUN mkdir -p checkpoints sample_images

# Copy checkpoints and sample images if they exist
# (use a wildcard so the build doesn't fail if empty)
COPY checkpoint[s]/ checkpoints/
COPY sample_image[s]/ sample_images/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
            "--server.port=8501", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
