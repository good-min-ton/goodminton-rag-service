# Image Embed Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up `goodminton-image-embed-service` — a new, isolated FastAPI microservice that loads SigLIP once at startup and turns an uploaded image into a 768-dim L2-normalized embedding vector for the cross-repo image-search feature.

**Architecture:** A single-file FastAPI app (`app/main.py`). A `lifespan` context loads `SiglipModel` + `SiglipImageProcessor` (`google/siglip-base-patch16-224`) once at process startup and stores them on `app.state`. `POST /embed/image` accepts one multipart file, decodes it behind a decompression-bomb + byte-cap guard, runs it through the model, L2-normalizes the output, and returns `{"embedding": [...]}`. `GET /health` reports readiness. The service exposes **only** these two routes — no batch/URL-fetch endpoint (no SSRF surface). It runs CPU-only, is Docker-packaged with the model baked in at build time, and is host-bound to `127.0.0.1:8001`.

**Tech Stack:** Python 3.11, `uv`, FastAPI, `transformers` (SigLIP), Pillow, CPU-only PyTorch (via the PyTorch CPU wheel index), pytest, Docker.

## Global Constraints

- **H1 — one response contract:** RAG `/search/image` returns `{"product_ids": ["42","17",…]}` — **strings, ranked by ascending distance**. Both entry points read `product_ids`. (No `results`/object-array drift.)
- **H2 — one multipart field name `file`** across all hops: FE→RAG uses `file`; RAG→embed-service uses `file`. (Matches existing `productsApi.uploadImage`.)
- **H4 — decompression-bomb guard:** set `Image.MAX_IMAGE_PIXELS` and catch `DecompressionBombError` in the embed-service decode path. Byte caps (RAG 8 MB outer, embed 10 MB) are **not** sufficient alone.
- **H6 — embed-service is host-bound to `127.0.0.1:8001`** and exposes ONLY `/embed/image` + `/health`. **No URL-fetch endpoint** (`/embed/images` removed) → **no SSRF surface**. RAG downloads Cloudinary images itself (trusted source) and POSTs bytes.
- **H11 — build-time internet:** the SigLIP model is baked into the embed-service image at build → `docker build` needs internet once; document (or `docker save/load` for offline defense).
- **H12 — memory: 3g** on the embed-service container (SigLIP RSS ~1.5–2.5 GB); `TORCH_NUM_THREADS` = container cpus.

---

### Task 1: Scaffold Repo & Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `tests/__init__.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: `app.main.app` — a `fastapi.FastAPI` instance (no routes yet). Task 2 adds `lifespan=` and routes to this same object.

- [ ] **Step 1: Initialize the repo**

```bash
mkdir -p /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service/app
mkdir -p /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service/tests
cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service
git init
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "goodminton-image-embed-service"
version = "0.1.0"
description = "SigLIP image-embedding microservice for Goodminton visual search"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "python-multipart>=0.0.9",
    "pillow>=10.4",
    "transformers>=4.44,<5",
    "torch>=2.3,<2.6",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "httpx>=0.25",
]

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv/
venv/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.idea/
.vscode/
*.log

# uv cache (uv.lock VẪN commit, chỉ ignore cache local)
.uv-cache/
```

- [ ] **Step 4: Install dependencies**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && uv sync`
Expected: `uv` resolves and installs all dependencies — including the CPU-only torch wheel pulled from the `pytorch-cpu` index — creates `uv.lock` and `.venv/`, exits 0. (This is the first, slowest install; torch + transformers are large downloads.)

- [ ] **Step 5: Write the failing smoke test**

```python
# tests/test_smoke.py
def test_app_importable():
    """Dependency-install sanity check: app.main must expose a FastAPI instance."""
    from fastapi import FastAPI

    from app.main import app

    assert isinstance(app, FastAPI)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && uv run pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'` (neither `app/__init__.py` nor `app/main.py` exist yet).

- [ ] **Step 7: Write the minimal implementation**

```python
# app/__init__.py
```

```python
# app/main.py
"""FastAPI entrypoint for the image-embedding microservice."""

from fastapi import FastAPI

app = FastAPI(
    title="Goodminton Image Embed Service",
    version="0.1.0",
    description="SigLIP image-embedding microservice for visual product search.",
)
```

```python
# tests/__init__.py
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && uv run pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service
git add pyproject.toml uv.lock .gitignore app tests
git commit -m "$(cat <<'EOF'
feat: scaffold image-embed-service (FastAPI + uv, CPU-only torch)

Empty FastAPI app skeleton with dependencies pinned; a smoke test
confirms the (heavy) ML dependency install and app.main import work
before any real endpoint is added.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Model Startup Lifespan & Health Check

**Files:**
- Modify: `app/main.py`
- Create: `tests/conftest.py`
- Test: `tests/test_health.py`

**Interfaces:**
- Consumes: `app.main.app` (Task 1).
- Produces: `app.state.model` (`transformers.SiglipModel`, eval mode) and `app.state.processor` (`transformers.SiglipImageProcessor`), loaded once at process startup — consumed by Task 3's `/embed/image` handler.
- Produces: `GET /health` → `{"status": "ok", "model_loaded": bool}`.
- Produces: `tests/conftest.py::client` — a session-scoped `fastapi.testclient.TestClient` fixture (runs the real lifespan once, loading the real model) — reused by Tasks 2–4's tests.

- [ ] **Step 1: Write the failing test + shared client fixture**

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """One TestClient for the whole test session.

    Entering it as a context manager runs the app's lifespan, which loads
    the real SigLIP model once (a few seconds) instead of once per test.
    """
    with TestClient(app) as c:
        yield c
```

```python
# tests/test_health.py
def test_health_returns_status_and_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && uv run pytest tests/test_health.py -v`
Expected: FAIL — 404 Not Found (no `/health` route defined yet).

- [ ] **Step 3: Write the minimal implementation**

```python
# app/main.py
"""FastAPI entrypoint for the image-embedding microservice."""

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from transformers import SiglipImageProcessor, SiglipModel

MODEL_NAME = "google/siglip-base-patch16-224"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # H12: TORCH_NUM_THREADS should match the container's cpu allocation;
    # falls back to the host's logical cpu count outside a container.
    threads = int(os.environ.get("TORCH_NUM_THREADS", os.cpu_count() or 1))
    torch.set_num_threads(threads)

    app.state.model = SiglipModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    app.state.model.eval()
    app.state.processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)

    yield

    app.state.model = None
    app.state.processor = None


app = FastAPI(
    title="Goodminton Image Embed Service",
    version="0.1.0",
    description="SigLIP image-embedding microservice for visual product search.",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && uv run pytest tests/test_health.py -v`
Expected: PASS. (First run downloads `google/siglip-base-patch16-224` weights from Hugging Face Hub — requires internet once; cached under `~/.cache/huggingface` afterward.)

- [ ] **Step 5: Commit**

```bash
cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service
git add app/main.py tests/conftest.py tests/test_health.py
git commit -m "$(cat <<'EOF'
feat: load SigLIP once at startup + add /health

Model + processor load in the FastAPI lifespan and are stashed on
app.state so the embed endpoint (next task) can reuse them without
reloading per request.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: POST /embed/image — Happy Path

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_embed_image.py`

**Interfaces:**
- Consumes: `app.state.model`, `app.state.processor` (Task 2); `tests/conftest.py::client` (Task 2).
- Produces: `POST /embed/image` (multipart field **`file`**, H2) → `{"embedding": list[float]}`, length 768, L2-normalized — consumed by Task 4 (adds guards to the same handler) and, cross-repo, by RAG's `EmbedClient`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed_image.py
import io
import math

from PIL import Image


def _solid_color_png_bytes(color=(200, 30, 30), size=(64, 64)) -> bytes:
    """Tiny in-memory PNG fixture — no network dependency."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_embed_image_returns_768_dim_normalized_vector(client):
    png_bytes = _solid_color_png_bytes()
    resp = client.post(
        "/embed/image",
        files={"file": ("swatch.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    embedding = resp.json()["embedding"]
    assert len(embedding) == 768
    norm = math.sqrt(sum(v * v for v in embedding))
    assert math.isclose(norm, 1.0, abs_tol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && uv run pytest tests/test_embed_image.py -v`
Expected: FAIL — 404 Not Found (no `/embed/image` route defined yet).

- [ ] **Step 3: Write the minimal implementation**

```python
# app/main.py — add these imports at the top, alongside the existing ones
import io

from fastapi import File, UploadFile
```

```python
# app/main.py — add below the /health route
@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    inputs = app.state.processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = app.state.model.get_image_features(**inputs)
    normalized = features / features.norm(p=2, dim=-1, keepdim=True)
    return {"embedding": normalized.squeeze(0).tolist()}
```

Also add the `Image` import from Pillow at the top of `app/main.py`:

```python
from PIL import Image
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && uv run pytest tests/test_embed_image.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service
git add app/main.py tests/test_embed_image.py
git commit -m "$(cat <<'EOF'
feat: add POST /embed/image happy path

Decodes the uploaded file, runs SigLIP get_image_features, and
L2-normalizes the result before returning it. Input guards (H4)
land in the next task.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: POST /embed/image — Input Guards (H4, H6)

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_embed_image_guards.py`

**Interfaces:**
- Consumes: `app.main.embed_image` (Task 3, same function, guards added inline); `tests/conftest.py::client` (Task 2).
- Produces: no new routes. Hardens the existing `/embed/image` route with a content-type check, a raw-byte cap, and a decompression-bomb guard (H4); asserts no `/embed/images` batch route exists (H6).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embed_image_guards.py
import io

from PIL import Image

from app import main as main_module


def _solid_color_png_bytes(size=(64, 64), color=(10, 200, 10)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_embed_image_rejects_non_image_content_type(client):
    resp = client.post(
        "/embed/image",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400


def test_embed_image_rejects_decompression_bomb(client, monkeypatch):
    # 40x40 = 1_600px vs MAX 1_000 -> 1.6x, i.e. INSIDE Pillow's 1x-2x
    # "warn-only" band where it does NOT raise DecompressionBombError.
    # This locks the explicit pixel-cap check (H4) — a catch-only guard fails here.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
    png_bytes = _solid_color_png_bytes(size=(40, 40))
    resp = client.post(
        "/embed/image",
        files={"file": ("big.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 400


def test_embed_image_rejects_oversize_upload(client, monkeypatch):
    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 10)
    png_bytes = _solid_color_png_bytes()
    resp = client.post(
        "/embed/image",
        files={"file": ("swatch.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 400


def test_no_batch_embed_endpoint_exists(client):
    """H6: no URL-fetch / batch endpoint — only /embed/image + /health exist."""
    resp = client.post(
        "/embed/images",
        files={"file": ("x.png", b"", "image/png")},
    )
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && uv run pytest tests/test_embed_image_guards.py -v`
Expected: FAIL — `test_embed_image_rejects_non_image_content_type` gets 200 instead of 400 (no content-type guard yet); `test_embed_image_rejects_decompression_bomb` gets 200 instead of 400 (no pixel-bomb guard yet); `test_embed_image_rejects_oversize_upload` errors with `AttributeError: <module 'app.main'> does not have the attribute 'MAX_UPLOAD_BYTES'` (cap doesn't exist yet); `test_no_batch_embed_endpoint_exists` already PASSes (route never existed) — the other three FAIL.

- [ ] **Step 3: Write the minimal implementation**

```python
# app/main.py — add near the top, after the MODEL_NAME constant
from PIL import UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50 MP decompression-bomb guard (H4)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB raw-byte cap (H4)
```

```python
# app/main.py — replace the embed_image body with the guarded version
@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="image exceeds max upload size")

    try:
        img = Image.open(io.BytesIO(data))
        # Pillow only *warns* (does not raise) between 1x and 2x MAX_IMAGE_PIXELS,
        # so enforce the pixel cap explicitly (H4) before decoding.
        if img.width * img.height > Image.MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=400, detail="image rejected by decompression-bomb guard"
            )
        img = img.convert("RGB")
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=400, detail="image rejected by decompression-bomb guard"
        )
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="invalid image data")

    inputs = app.state.processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = app.state.model.get_image_features(**inputs)
    normalized = features / features.norm(p=2, dim=-1, keepdim=True)
    return {"embedding": normalized.squeeze(0).tolist()}
```

Also add `HTTPException` to the existing `from fastapi import ...` line:

```python
from fastapi import File, HTTPException, UploadFile
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && uv run pytest tests/test_embed_image_guards.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Run the full suite**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && uv run pytest -v`
Expected: PASS (all tests across `test_smoke.py`, `test_health.py`, `test_embed_image.py`, `test_embed_image_guards.py`)

- [ ] **Step 6: Commit**

```bash
cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service
git add app/main.py tests/test_embed_image_guards.py
git commit -m "$(cat <<'EOF'
feat: guard /embed/image against bombs, bad content-type, oversize uploads

Adds the H4 decompression-bomb guard (Image.MAX_IMAGE_PIXELS +
DecompressionBombError), a content-type check, and a raw-byte cap.
Also pins H6: confirms no /embed/images batch endpoint exists.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Dockerfile & Compose Service (H6, H11, H12)

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

**Interfaces:**
- Consumes: `pyproject.toml` + `uv.lock` (Task 1); `app/main.py` (Tasks 1–4).
- Produces: a buildable image `goodminton-image-embed-service:local` exposing port 8001, and a standalone compose service `embed-service` bound to `127.0.0.1:8001` — the integration point RAG's `EmbedClient` (`http://localhost:8001`, out of scope for this repo) will call.

This is infra, not application code — there's no pytest for a Docker build, so verification uses `docker compose config` (fast, no network) and `docker build` (slow, real build).

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# Image-embedding microservice — SigLIP (see design spec §1 embed-service).
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Cache deps layer riêng
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy code
COPY . .
RUN uv sync --frozen --no-dev

# H11: bake the SigLIP model into the image at build time so the container
# never needs internet access at runtime. This means `docker build` needs
# internet once; for fully offline environments, build here and transfer
# the image with `docker save` / `docker load`.
RUN uv run python -c "\
from transformers import SiglipImageProcessor, SiglipModel; \
SiglipModel.from_pretrained('google/siglip-base-patch16-224', use_safetensors=True); \
SiglipImageProcessor.from_pretrained('google/siglip-base-patch16-224')"

# H11: guarantee no runtime Hub connectivity — serve the baked model from cache only.
# (Also keeps startup fast: no Hub connectivity check to time out on.)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 2: Write the compose service**

```yaml
# Standalone compose for the image-embed microservice (local/dev use).
# Host-bound to 127.0.0.1 only (H6) — no external network exposure; RAG
# (running native on the same host) reaches it via http://localhost:8001.
#
#   docker compose up -d --build
services:
  embed-service:
    build:
      context: .
    image: goodminton-image-embed-service:local
    container_name: goodminton-image-embed-service
    restart: unless-stopped
    ports:
      - "127.0.0.1:8001:8001"
    mem_limit: 3g # H12
    cpus: 2 # H12: bound CPU so TORCH_NUM_THREADS matches the allocation
    environment:
      TORCH_NUM_THREADS: "2" # H12: matches the cpus limit above
    healthcheck:
      # slim base image has no curl; use python (guaranteed present)
      test: [ "CMD-SHELL", "python -c \"import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://localhost:8001/health').status==200 else 1)\"" ]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 60s # model load can take a few seconds on cold start
```

- [ ] **Step 3: Validate compose syntax**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && docker compose config`
Expected: exits 0 and prints the resolved `embed-service` service block (confirms valid YAML + compose schema, no build/network needed).

- [ ] **Step 4: Build the image**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && docker build -t goodminton-image-embed-service:local .`
Expected: build succeeds (exit 0); the final layers show the `uv sync` installs and the model-baking `RUN` step completing without error. (Needs internet for base image, dependency wheels, and the SigLIP weights — H11; can take several minutes.)

- [ ] **Step 5: Commit**

```bash
cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service
git add Dockerfile docker-compose.yml
git commit -m "$(cat <<'EOF'
feat: add Dockerfile + compose service, model baked at build

Mirrors goodminton-rag-service's uv-based Dockerfile pattern. Bakes
the SigLIP model at build time (H11) so the container never needs
runtime internet. Compose service is host-bound to 127.0.0.1:8001
(H6) with a 3g memory cap (H12).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-image-embed-service && uv run pytest`
Expected: PASS — covers Tasks 1–4 (smoke, health, happy-path embed, guards). Task 5's Dockerfile/compose are verified separately via `docker compose config` and `docker build` (Step 3–4 above); they aren't part of the pytest suite.
