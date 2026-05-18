# Goodminton RAG Service

Trợ lý RAG tư vấn sản phẩm cầu lông cho Goodminton Shop.

Thiết kế tổng thể: xem [`goodminton-shop-api/docs/rag-chatbot-guide.md`](../goodminton-shop-api/docs/rag-chatbot-guide.md).

## Trạng thái

- [x] Phase 1 — Foundation (pgvector, RabbitMQ, Ollama) trong repo `goodminton-shop-api`
- [x] Phase 2 — Static docs + index script
- [x] **Phase 3 — FastAPI app + /chat endpoint + retrieval** ← bạn đang xem
- [ ] Phase 4 — RabbitMQ consumer cho product sync
- [ ] Phase 5 — Tool calling cho real-time data (price, stock)
- [ ] Phase 6 — NextJS UI integration

## Cấu trúc

```
rag-service/
├── app/
│   ├── main.py                   # FastAPI entrypoint + lifespan
│   ├── routers/
│   │   └── chat.py               # POST /chat
│   ├── services/
│   │   ├── embedding.py          # Ollama bge-m3 wrapper
│   │   ├── retrieval.py          # pgvector similarity search
│   │   └── llm.py                # Ollama Qwen chat wrapper
│   ├── core/
│   │   ├── config.py             # Pydantic settings (env)
│   │   ├── prompts.py            # System prompt
│   │   └── db.py                 # asyncpg pool
│   └── models/
│       └── schemas.py            # Pydantic request/response
├── data/
│   └── static_docs/              # Lớp A — bộ tối thiểu
│       ├── 01-chinh-sach-bao-hanh.md
│       ├── 02-chinh-sach-doi-tra.md
│       ├── 03-thong-tin-shop.md
│       └── 04-huong-dan-can-day-vot.md
├── scripts/
│   └── index_static_docs.py      # Embed + INSERT vào kb_chunks (chạy 1 lần)
├── Dockerfile                    # FastAPI image (default uvicorn)
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

## Phase 3 — Run FastAPI service

### Build image

```bash
cd /path/to/goodminton-rag-service
docker build -t goodminton-rag-service:latest .
```

### Run

```bash
docker run -d \
  --name goodminton-rag \
  --network goodminton_default \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://USER:PASS@postgres:5432/goodminton" \
  -e OLLAMA_URL="http://ollama:11434" \
  goodminton-rag-service:latest
```

### Test

```bash
# Health
curl http://localhost:8000/health
# {"status":"ok"}

# Chat — câu về policy
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Vợt mua ở shop bảo hành mấy tháng?",
    "chat_history": []
  }'

# Chat với history
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Còn lực căng dây nào phù hợp người mới không?",
    "chat_history": [
      {"role": "user", "content": "Tôi mới chơi cầu lông"},
      {"role": "assistant", "content": "Chào bạn! Bạn cần tư vấn về vợt, giày, hay cách chơi?"}
    ]
  }'
```

Response format:

```json
{
  "answer": "...",
  "sources": [
    {"doc_type": "static", "source_id": "01-chinh-sach-bao-hanh.md"},
    {"doc_type": "static", "source_id": "04-huong-dan-can-day-vot.md"}
  ]
}
```

### Docs Swagger

Mở `http://localhost:8000/docs` để xem OpenAPI spec + thử endpoint qua UI.
