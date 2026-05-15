# ── Stage 1: Build ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies required for compilation of Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ───────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Sentence-transformers model cache inside the image layer
    HF_HOME=/app/.cache/huggingface \
    # Tesseract binary path (matches TESSERACT_CMD default in config.py)
    TESSERACT_CMD=/usr/bin/tesseract

# Runtime system dependencies:
#   tesseract-ocr   — local OCR fallback
#   poppler-utils   — pdf2image / PyMuPDF rasterisation
#   libgl1          — OpenCV headless runtime (libgl1-mesa-glx on older distros)
#   libglib2.0-0    — OpenCV runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy all source files (excluding what's in .dockerignore)
COPY . .

# Create writable runtime directories
RUN mkdir -p data/chroma_db data/sample_inputs sample_outputs .cache/huggingface

# Expose FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default: run the FastAPI server.
# Override in docker-compose.yml for the Streamlit service.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
