"""
Index static docs (markdown) vào bảng kb_chunks (pgvector).

Pipeline:
1. Đọc tất cả files trong data/static_docs/*.md
2. Split mỗi file thành chunks (500 chars, overlap 50)
3. Embed mỗi chunk qua Ollama bge-m3 (dim 1024)
4. DELETE chunks cũ (doc_type='static') rồi INSERT mới — re-index toàn bộ.

Yêu cầu:
- pgvector extension đã được cài (V7 migration của shop-api)
- Ollama đang chạy với model bge-m3 đã pull
- Env: DATABASE_URL, OLLAMA_URL

Chạy:
  python scripts/index_static_docs.py
"""
import asyncio
import os
import sys
from pathlib import Path

import asyncpg
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pgvector.asyncpg import register_vector

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
DOCS_DIR = Path(__file__).parent.parent / "data" / "static_docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


async def embed(client: httpx.AsyncClient, text: str) -> list[float]:
    """Gọi Ollama /api/embeddings để lấy vector."""
    r = await client.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()["embedding"]


async def main() -> None:
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL env required", file=sys.stderr)
        sys.exit(1)

    if not DOCS_DIR.exists():
        print(f"ERROR: docs dir not found: {DOCS_DIR}", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        print(f"WARNING: no .md files in {DOCS_DIR}")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    conn = await asyncpg.connect(DATABASE_URL)
    await register_vector(conn)

    async with httpx.AsyncClient() as http:
        # Re-index strategy: xoá hết static chunks rồi insert lại.
        # Phù hợp scale < 1000 chunks. Lớn hơn cần diff-based.
        await conn.execute("DELETE FROM kb_chunks WHERE doc_type = 'static'")
        print("Cleared old static chunks\n")

        total = 0
        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8")
            chunks = splitter.split_text(text)
            source_id = md_file.name

            for idx, chunk in enumerate(chunks):
                embedding = await embed(http, chunk)
                await conn.execute(
                    """
                    INSERT INTO kb_chunks
                        (doc_type, source_id, chunk_index, content, embedding)
                    VALUES ('static', $1, $2, $3, $4)
                    """,
                    source_id,
                    idx,
                    chunk,
                    embedding,
                )

            print(f"  {md_file.name}: {len(chunks)} chunks")
            total += len(chunks)

    await conn.close()
    print(f"\nDone. Total: {total} chunks indexed.")


if __name__ == "__main__":
    asyncio.run(main())
