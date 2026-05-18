# Phase 3 — FastAPI service.
# Index script vẫn chạy được bằng: docker run ... <image> uv run python scripts/index_static_docs.py
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Cache deps layer riêng
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy code
COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000

# Default: chạy FastAPI app
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
