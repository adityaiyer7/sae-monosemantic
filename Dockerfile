FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System deps for duckdb and pyarrow native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (layer-cached if pyproject.toml/uv.lock unchanged)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-install-package torch

# Copy project source — app/ and src/ needed; heavy dirs are excluded via .dockerignore
COPY app/ ./app/
COPY src/ ./src/

# Point HF cache into /app so it survives the user permission switch below
ENV HF_HOME=/app/.cache/huggingface

# Pre-download GPT-2 tokenizer so first startup doesn't fetch it from HuggingFace
RUN .venv/bin/python -c "from transformers import GPT2Tokenizer; GPT2Tokenizer.from_pretrained('gpt2')"

# HF Spaces run containers as UID 1000 — create that user and hand over ownership
RUN useradd -m -u 1000 -s /bin/bash user && chown -R user:user /app
USER user

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD [".venv/bin/streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
