# Goodminton RAG Service

Trợ lý RAG tư vấn sản phẩm cầu lông cho Goodminton Shop.

Thiết kế tổng thể: xem [`goodminton-shop-api/docs/rag-chatbot-guide.md`](../goodminton-shop-api/docs/rag-chatbot-guide.md).

## Trạng thái

- [x] Phase 1 — Foundation (pgvector, RabbitMQ, Ollama) trong repo `goodminton-shop-api`
- [x] **Phase 2 — Static docs + index script** ← bạn đang xem
- [ ] Phase 3 — FastAPI app + /chat endpoint + retrieval
- [ ] Phase 4 — RabbitMQ consumer cho product sync
- [ ] Phase 5 — Tool calling cho real-time data (price, stock)
- [ ] Phase 6 — NextJS UI integration

## Cấu trúc

```
rag-service/
├── data/
│   └── static_docs/              # Lớp A — bộ tối thiểu
│       ├── 01-chinh-sach-bao-hanh.md
│       ├── 02-chinh-sach-doi-tra.md
│       ├── 03-thong-tin-shop.md
│       └── 04-huong-dan-can-day-vot.md
├── scripts/
│   └── index_static_docs.py      # Embed + INSERT vào kb_chunks
├── Dockerfile                    # Build image cho indexer (Phase 3 sẽ mở rộng)
├── pyproject.toml                # uv-managed deps (PEP 621)
├── uv.lock                       # Lock file — commit cùng pyproject.toml
└── .env.example
```

## Setup local dev (dùng uv)

Cài uv (nếu chưa có):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Generate lock file + install deps:

```bash
cd /path/to/goodminton-rag-service
uv sync          # Tạo .venv/, cài deps từ pyproject.toml, tạo uv.lock nếu chưa có
```

Chạy script (uv tự kích hoạt venv):

```bash
uv run python scripts/index_static_docs.py
```

Thêm/xoá dependency:

```bash
uv add httpx               # thêm runtime dep
uv add --dev ruff          # thêm dev dep
uv remove httpx            # gỡ
```

## Phase 2 — Chạy index script

### Tiền đề

Trên VPS (hoặc local dev):

- Stack `goodminton-shop-api` đang chạy với compose network `goodminton_default`
- pgvector extension installed (qua Flyway V7)
- Ollama đã pull model `bge-m3`

### Build image

```bash
cd /path/to/goodminton-rag-service
docker build -t goodminton-rag-indexer:latest .
```

### Chạy index

```bash
docker run --rm \
  --network goodminton_default \
  -e DATABASE_URL="postgresql://USER:PASS@postgres:5432/goodminton" \
  -e OLLAMA_URL="http://ollama:11434" \
  -v "$(pwd)/data/static_docs:/app/data/static_docs:ro" \
  goodminton-rag-indexer:latest
```

Output mong đợi:

```
Cleared old static chunks

  01-chinh-sach-bao-hanh.md: 3 chunks
  02-chinh-sach-doi-tra.md: 4 chunks
  03-thong-tin-shop.md: 3 chunks
  04-huong-dan-can-day-vot.md: 5 chunks

Done. Total: 15 chunks indexed.
```

### Verify trong Postgres

```bash
docker exec -it goodminton-postgres psql -U USER -d goodminton -c \
  "SELECT doc_type, source_id, COUNT(*) FROM kb_chunks GROUP BY doc_type, source_id ORDER BY source_id;"
```

### Test retrieval (similarity search)

```bash
docker exec -it goodminton-postgres psql -U USER -d goodminton << 'SQL'
WITH q AS (
  -- Embedding của query (sẽ làm trong Phase 3 — đây chỉ là placeholder)
  SELECT embedding FROM kb_chunks WHERE doc_type='static' LIMIT 1
)
SELECT source_id, chunk_index, LEFT(content, 100)
FROM kb_chunks, q
WHERE doc_type='static'
ORDER BY kb_chunks.embedding <=> q.embedding
LIMIT 3;
SQL
```

## Cập nhật nội dung docs

Sửa file `.md` trong `data/static_docs/` → rebuild image → chạy lại lệnh `docker run` trên. Script tự xoá chunks cũ và insert mới.

Phase 3 sẽ tự động hoá: image rag-service start lên → init container chạy index trước khi FastAPI listen.
