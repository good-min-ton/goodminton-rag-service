# Phase 2 — image dùng để chạy index script.
# Phase 3 sẽ mở rộng cho FastAPI service.
#
# Dùng official uv image (Astral) — base có sẵn Python + uv CLI, nhanh hơn pip.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Cache dependencies layer riêng (tận dụng Docker layer cache):
# Chỉ copy pyproject.toml + uv.lock → install deps → sau đó mới copy code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy code
COPY . .

# Install project (không có gì thêm ngoài đã cài, nhưng đúng convention)
RUN uv sync --frozen --no-dev

# Default: chạy index script. Override bằng docker run ... <cmd>
CMD ["uv", "run", "python", "scripts/index_static_docs.py"]
