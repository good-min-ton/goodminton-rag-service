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
#
# --forwarded-allow-ips: bắt buộc từ khi có gateway. Hai bộ giới hạn tần suất
# trong search.py và chat.py đếm theo request.client.host, tức địa chỉ TCP của
# bên gọi - từ nay luôn là container gateway, giống hệt nhau với mọi khách. Hai
# hạn mức vốn là "mỗi IP" lặng lẽ biến thành "toàn hệ thống": 20 lượt tìm ảnh và
# 30 lượt chat mỗi phút chia chung cho tất cả người dùng, nên một người là đủ
# khoá cả web. Không có lỗi nào hiện ra, chỉ có 429 cho người vô can.
#
# Gateway đã gửi X-Forwarded-For sẵn. --proxy-headers cũng đã bật mặc định, thứ
# thiếu là danh sách địa chỉ được tin: mặc định chỉ có 127.0.0.1, mà gateway nằm
# ở một IP khác trong mạng Docker nên header bị bỏ qua. Ghi cả hai cờ cho rõ ý.
#
# "*" an toàn vì sau khi gỡ ánh xạ cổng ra host thì chỉ gateway mới nối tới được
# service này; trước đó thì cổng vẫn mở nên cũng chẳng có gì để giả mạo thêm.
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
