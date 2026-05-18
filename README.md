# Goodminton RAG Service

RAG assistant for Goodminton Shop — helps customers learn about products, get personalized recommendations, and look up real-time info (Phase 5+).

Design spec: see [`goodminton-shop-api/docs/rag-chatbot-guide.md`](../goodminton-shop-api/docs/rag-chatbot-guide.md).

## Status

- [x] Phase 1 — Foundation (pgvector, RabbitMQ, Ollama) — lives in `goodminton-shop-api` repo
- [x] Phase 2 — Static docs + index script
- [x] **Phase 3 — FastAPI app + `/chat` endpoint + retrieval** ← current
- [ ] Phase 4 — RabbitMQ consumer for product sync
- [ ] Phase 5 — Tool calling for real-time data (price, stock)
- [ ] Phase 6 — NextJS UI integration

## Repository layout

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
│   │   ├── config.py             # Pydantic settings (env-driven)
│   │   ├── prompts.py            # System prompt
│   │   └── db.py                 # asyncpg pool factory
│   └── models/
│       └── schemas.py            # Pydantic request/response models
├── data/
│   └── static_docs/              # Layer A — minimal policy/info docs
│       ├── 01-chinh-sach-bao-hanh.md
│       ├── 02-chinh-sach-doi-tra.md
│       ├── 03-thong-tin-shop.md
│       └── 04-huong-dan-can-day-vot.md
├── scripts/
│   └── index_static_docs.py      # Embed + UPSERT into kb_chunks
├── Dockerfile                    # FastAPI image (default: uvicorn)
├── pyproject.toml                # uv-managed deps (PEP 621)
├── uv.lock                       # Lock file — commit alongside pyproject.toml
└── .env.example
```

## Local development (uv)

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sync dependencies into a local `.venv`:

```bash
cd /path/to/goodminton-rag-service
uv sync          # Creates .venv/, installs from pyproject.toml, generates uv.lock if missing
```

Run any script through uv (it activates the venv automatically):

```bash
uv run python scripts/index_static_docs.py
uv run uvicorn app.main:app --reload
```

Manage dependencies:

```bash
uv add httpx              # add runtime dep
uv add --dev ruff         # add dev dep
uv remove httpx           # remove
uv lock --upgrade         # upgrade all deps to latest compatible
```

Before committing, run lint + format:

```bash
uv run ruff format .
uv run ruff check .
```

## Phase 2 — Index static docs

### Prerequisites

- The `goodminton-shop-api` stack is running, exposing the compose network `goodminton_default`.
- pgvector extension is installed via Flyway migration V7.
- Ollama is running with the `bge-m3` embedding model pulled.

### Build the image

```bash
cd /path/to/goodminton-rag-service
docker build -t goodminton-rag-service:latest .
```

### Run the indexer

```bash
docker run --rm \
  --network goodminton_default \
  -e DATABASE_URL="postgresql://USER:PASS@postgres:5432/goodminton" \
  -e OLLAMA_URL="http://ollama:11434" \
  goodminton-rag-service:latest \
  uv run python scripts/index_static_docs.py
```

Expected output:

```
Cleared old static chunks

  01-chinh-sach-bao-hanh.md: 6 chunks
  02-chinh-sach-doi-tra.md: 5 chunks
  03-thong-tin-shop.md: 5 chunks
  04-huong-dan-can-day-vot.md: 6 chunks

Done. Total: 22 chunks indexed.
```

### Verify in Postgres

```bash
docker exec -it goodminton-postgres psql -U USER -d goodminton -c \
  "SELECT doc_type, source_id, COUNT(*) FROM kb_chunks GROUP BY doc_type, source_id ORDER BY source_id;"
```

### Updating docs content

Edit any `.md` file under `data/static_docs/`, rebuild the image, and re-run the indexer command above. The script deletes existing `doc_type='static'` chunks and re-inserts.

In CD, the `reindex-static-docs` job runs this automatically after every successful deploy.

## Phase 3 — Run the FastAPI service

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

# Chat — policy question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Vợt mua ở shop bảo hành mấy tháng?",
    "chat_history": []
  }'

# Chat with history
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

Response shape:

```json
{
  "answer": "...",
  "sources": [
    {"doc_type": "static", "source_id": "01-chinh-sach-bao-hanh.md"},
    {"doc_type": "static", "source_id": "04-huong-dan-can-day-vot.md"}
  ]
}
```

### OpenAPI UI

Open `http://localhost:8000/docs` to browse the spec and try requests interactively.

## CI/CD

- **CI** (`.github/workflows/ci.yml`) — runs on PRs and pushes to `main`. Lints with ruff and verifies the Docker image builds.
- **CD** (`.github/workflows/cd.yml`) — runs on pushes to `main`. Three jobs:
  1. `build` — builds and pushes the Docker image to Docker Hub.
  2. `deploy` — on the self-hosted runner, pulls the new image and recreates the `rag-service` container via compose.
  3. `reindex-static-docs` — runs `index_static_docs.py` against the live stack to refresh chunks.

The `rag-service` service is declared in `goodminton-shop-api/docker-compose.prod.yml` and runs alongside the rest of the stack on the VPS.

Required secrets (org-level `good-min-ton`):

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD` (Docker Hub access token)
