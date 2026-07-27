# Image Search UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax.

**Goal:** Let a shopper find visually-similar products from a photo through BOTH the storefront search bar (📷 → `/products` grid) and the chatbot (attach/paste → product cards), reusing the existing results grid and `ProductSourceCards`.

**Architecture:** A shared `searchApi.searchByImage(file)` POSTs the image (multipart field `file`, H2) to `NEXT_PUBLIC_RAG_API_URL/search/image` and reads back `{ product_ids: string[] }` (H1). Entry 1 (header) downscales the image client-side, calls `searchByImage` then `searchApi.listItemsByIds(ids)` (shop-api hydration), stashes the hydrated list in a small zustand store, and navigates to `/products?mode=image` which renders it via the existing search-result card grid (`ProductGrid` untouched). Entry 2 (chatbot) calls `searchByImage` directly and renders the returned ids through the existing `ProductSourceCards` with a fixed caption; the chat LLM is bypassed.

**Tech Stack:** Next.js 16.2.4, React 19, TypeScript 5 (strict), TanStack Query, zustand, Tailwind v4, lucide-react. UI repo root: `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui` (a SIBLING of this RAG repo).

## Global Constraints
- **H1 — one response contract:** `/search/image` returns `{"product_ids": ["42","17",…]}` — strings, ranked ascending distance. Both entry points read `product_ids`. No `results`/object-array drift.
- **H2 — one multipart field name `file`:** the FE→RAG upload uses form field name `file` (mirrors `productsApi.uploadImage`'s `fd.append("file", …)`).
- **H8 — visibility is server-side:** hydration goes through shop-api `GET /api/products/list-items?ids=…`, which filters `is_visible` and preserves requested order. The UI does not re-filter or re-order.
- **Client-side downscale (review #8):** before upload, validate MIME (`image/jpeg,image/png,image/webp`) and downscale via canvas to keep the payload under ~2 MB.
- **Text search unchanged:** the `?q=` path in `header-search.tsx` and `products/page.tsx` and all of `searchApi.products*` stay exactly as-is.
- **ChatRequest to RAG unchanged:** the `/chat` request body stays `{ message, chat_history }`. Image search is a SEPARATE call to `/search/image`; it does not add fields to `ChatRequest` and does not touch `components/chatbot/api.ts:sendChat`.
- **`ProductGrid` unchanged:** it takes `Product[]`; image results are `ProductListItem[]`, so they render through the existing `SearchResultCard` grid in `products/page.tsx`, not through `ProductGrid`.
- **No test runner:** `package.json` scripts are only `dev`/`build`/`start`. The verification gate is `npx tsc --noEmit` (exit 0) plus concrete manual checks. Do NOT add or invoke any test runner.

---

### Task 1: Shared image utilities — `searchApi` contract + client-side downscale
**Files:**
- Create `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui/lib/image-downscale.ts`
- Modify `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui/lib/api/search.ts` (imports at lines 1–10; `searchApi` object spans lines 50–77, closing `};` at line 77)

**Interfaces:**
- Consumes: RAG `POST /search/image` (multipart `file` → `{ product_ids: string[] }`, H1/H2); shop-api `GET /api/products/list-items?ids=1,2,3` → `ProductListItem[]` (H8); `process.env.NEXT_PUBLIC_RAG_API_URL`; existing `api.get` / `buildQuery` from `@/lib/api`.
- Produces: `downscaleImage(file, maxDimension?, maxBytes?) => Promise<Blob>`, `makeThumbnailDataUrl(file, maxDimension?) => Promise<string>` (from `lib/image-downscale.ts`); `searchApi.searchByImage(file: File | Blob) => Promise<ImageSearchResponse>`, `searchApi.listItemsByIds(ids: number[]) => Promise<ProductListItem[]>`, and exported `interface ImageSearchResponse { product_ids: string[] }`.

- [ ] **Step 0: Create the feature branch first** The UI repo is on `main` — branch before any commit (repo rule; also makes Task 4's `git diff main..HEAD` verification meaningful instead of empty).
  ```bash
  cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && git switch -c feat/image-search-ui
  ```

- [ ] **Step 1: Create the downscale util** Write `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui/lib/image-downscale.ts`:
  ```ts
  /** Client-side image downscale/recompress so uploads stay small (< ~2 MB).
   *  Shared by the storefront 📷 search and the chatbot image attach. */

  async function canvasToBlob(
    canvas: HTMLCanvasElement,
    type: string,
    quality: number,
  ): Promise<Blob> {
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("toBlob failed"))),
        type,
        quality,
      );
    });
  }

  function scaledCanvas(
    bitmap: ImageBitmap,
    maxDimension: number,
  ): HTMLCanvasElement {
    const scale = Math.min(
      1,
      maxDimension / Math.max(bitmap.width, bitmap.height),
    );
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D context unavailable");
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  /** Downscale to <= maxDimension px on the long edge and recompress to JPEG
   *  under maxBytes (drops quality in steps; floor 0.4). */
  export async function downscaleImage(
    file: File | Blob,
    maxDimension = 1024,
    maxBytes = 2 * 1024 * 1024,
  ): Promise<Blob> {
    const bitmap = await createImageBitmap(file);
    try {
      const canvas = scaledCanvas(bitmap, maxDimension);
      let quality = 0.9;
      let blob = await canvasToBlob(canvas, "image/jpeg", quality);
      while (blob.size > maxBytes && quality > 0.4) {
        quality -= 0.15;
        blob = await canvasToBlob(canvas, "image/jpeg", quality);
      }
      return blob;
    } finally {
      bitmap.close();
    }
  }

  /** Small JPEG data URL for showing the sent image in chat history. */
  export async function makeThumbnailDataUrl(
    file: File | Blob,
    maxDimension = 256,
  ): Promise<string> {
    const bitmap = await createImageBitmap(file);
    try {
      const canvas = scaledCanvas(bitmap, maxDimension);
      return canvas.toDataURL("image/jpeg", 0.7);
    } finally {
      bitmap.close();
    }
  }
  ```

- [ ] **Step 2: Add the RAG base URL + response type to `search.ts`** After the import block (after line 10 of `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui/lib/api/search.ts`), insert:
  ```ts
  const RAG_API_URL =
    process.env.NEXT_PUBLIC_RAG_API_URL?.replace(/\/$/, "") ?? "";

  /** RAG /search/image response — H1 contract: ranked product ids as strings. */
  export interface ImageSearchResponse {
    product_ids: string[];
  }
  ```

- [ ] **Step 3: Add `searchByImage` + `listItemsByIds` to the `searchApi` object** Insert these two methods just before the closing `};` of the `searchApi` object (currently line 77 of `search.ts`, immediately after the `stores(query) { … },` method):
  ```ts
    /** Visual search: POST the image to RAG /search/image (multipart field
     *  `file` — H2) and read back ranked product ids (strings — H1). Hits the
     *  RAG service directly (raw JSON, not the shop-api {code,result} envelope,
     *  mirroring components/chatbot/api.ts). */
    async searchByImage(file: File | Blob): Promise<ImageSearchResponse> {
      if (!RAG_API_URL) {
        throw new Error("Image search chưa được cấu hình.");
      }
      const fd = new FormData();
      fd.append("file", file, "upload.jpg"); // H2: field name `file`
      const res = await fetch(`${RAG_API_URL}/search/image`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        throw new Error(`Image search failed (${res.status})`);
      }
      return res.json() as Promise<ImageSearchResponse>;
    },

    /** Hydrate RAG image-search ids → list items via shop-api. Order + is_visible
     *  are enforced server-side (H8); the UI renders the array as-received. */
    listItemsByIds(ids: number[]): Promise<ProductListItem[]> {
      if (ids.length === 0) return Promise.resolve([]);
      return api.get<ProductListItem[]>(
        `/api/products/list-items?ids=${ids.join(",")}`,
      );
    },
  ```
  (`ProductListItem`, `api`, and `buildQuery` are already imported at the top of `search.ts` — no new imports needed.)

- [ ] **Step 4: Type gate** Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && npx tsc --noEmit`  Expected: exits 0 with no output (no errors). If it fails with `Cannot find name 'createImageBitmap'` or `toBlob`, confirm `tsconfig.json` `lib` includes `dom` (it does) — a failure here means a typo, not a missing lib.

- [ ] **Step 5: Manual verification (no UI yet — contract smoke test)** Confirm the module surface exists and is importable:
  - Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && grep -n "searchByImage\|listItemsByIds\|ImageSearchResponse" lib/api/search.ts && grep -n "export async function downscaleImage\|export async function makeThumbnailDataUrl" lib/image-downscale.ts`
  - Expected: `searchByImage`, `listItemsByIds`, `ImageSearchResponse` each print once in `search.ts`; both exported functions print once in `image-downscale.ts`.

- [ ] **Step 6: Commit**
  ```bash
  cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && \
  git add lib/image-downscale.ts lib/api/search.ts && \
  git commit -m "feat(search): add image-search API contract + client downscale util

  searchApi.searchByImage POSTs multipart 'file' (H2) to RAG /search/image and
  reads {product_ids:[strings]} (H1); listItemsByIds hydrates via shop-api
  list-items (visible+ordered server-side, H8). Shared canvas downscale keeps
  uploads < ~2MB.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 2: Entry 1 — storefront 📷 search bar → `/products` image-results mode
**Files:**
- Create `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui/store/image-search-store.ts`
- Modify `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui/components/storefront/header-search.tsx` (imports lines 1–12; component state lines 14–20; `submit` ends line 67; trigger `<button>` lines 71–78; wrapper `<div>` opens line 70)
- Modify `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui/app/(storefront)/products/page.tsx` (imports lines 3–16; `ProductsListing` lines 43–76; return ternary lines 62–74; `SearchResultCard` component lines 275–314; `Spinner`/`EmptyState` already imported lines 11–12)

**Interfaces:**
- Consumes: `searchApi.searchByImage`, `searchApi.listItemsByIds` (Task 1); `downscaleImage` (Task 1); `ProductListItem` from `@/types/api`; `useRouter` (already imported in `header-search.tsx`).
- Produces: `useImageSearchStore` zustand store (`status: "idle"|"loading"|"success"|"error"`, `results: ProductListItem[]`, `error: string | null`, `start()`, `succeed(results)`, `fail(error)`, `reset()`); a camera button + hidden file input in `header-search.tsx`; an `ImageSearchResults` sub-component + `mode=image` branch in `products/page.tsx`.

- [ ] **Step 1: Create the zustand store** Write `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui/store/image-search-store.ts` (mirrors the `create<…>()` pattern in `store/auth-store.ts`):
  ```ts
  "use client";

  import { create } from "zustand";
  import type { ProductListItem } from "@/types/api";

  type ImageSearchStatus = "idle" | "loading" | "success" | "error";

  interface ImageSearchState {
    status: ImageSearchStatus;
    results: ProductListItem[];
    error: string | null;
    /** Called by the header before navigating: reset + enter loading. */
    start: () => void;
    succeed: (results: ProductListItem[]) => void;
    fail: (error: string) => void;
    reset: () => void;
  }

  export const useImageSearchStore = create<ImageSearchState>((set) => ({
    status: "idle",
    results: [],
    error: null,
    start: () => set({ status: "loading", results: [], error: null }),
    succeed: (results) => set({ status: "success", results, error: null }),
    fail: (error) => set({ status: "error", error, results: [] }),
    reset: () => set({ status: "idle", results: [], error: null }),
  }));
  ```

- [ ] **Step 2: Wire the camera button + handler into `header-search.tsx`** Make three edits.
  (a) Replace the import at line 7 (`import { Search, X } from "lucide-react";`) with:
  ```ts
  import { Camera, Search, X } from "lucide-react";
  ```
  and add these two imports directly below the existing `import { searchApi } from "@/lib/api/search";` (line 8):
  ```ts
  import { downscaleImage } from "@/lib/image-downscale";
  import { useImageSearchStore } from "@/store/image-search-store";
  ```
  (b) Add a file-input ref beside the existing refs (after line 20, `const inputRef = useRef<HTMLInputElement>(null);`):
  ```ts
  const fileInputRef = useRef<HTMLInputElement>(null);
  ```
  and add this handler immediately after the `submit` function (after line 67, its closing `}`):
  ```ts
  async function onImageSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file later
    if (!file) return;

    const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];
    const store = useImageSearchStore.getState();
    setOpen(false);
    setQuery("");
    store.start();
    router.push("/products?mode=image");

    if (!ACCEPTED.includes(file.type)) {
      store.fail("Định dạng ảnh không hỗ trợ. Dùng JPG, PNG hoặc WEBP.");
      return;
    }

    try {
      const downscaled = await downscaleImage(file);
      const { product_ids } = await searchApi.searchByImage(downscaled);
      const ids = product_ids
        .map(Number)
        .filter((n) => Number.isInteger(n) && n > 0);
      const items = ids.length > 0 ? await searchApi.listItemsByIds(ids) : [];
      store.succeed(items);
    } catch {
      store.fail("Không tìm được sản phẩm từ ảnh. Thử lại nhé.");
    }
  }
  ```
  (c) Add the camera trigger + hidden input directly after the existing search trigger `</button>` (after line 78), still inside the `wrapperRef` div:
  ```tsx
  <button
    type="button"
    onClick={() => fileInputRef.current?.click()}
    className="rounded-lg p-2 text-stone-600 hover:bg-stone-100 hover:text-stone-900"
    aria-label="Tìm bằng hình ảnh"
  >
    <Camera size={20} />
  </button>
  <input
    ref={fileInputRef}
    type="file"
    accept="image/jpeg,image/png,image/webp"
    className="hidden"
    onChange={onImageSelected}
  />
  ```

- [ ] **Step 3: Add the `mode=image` branch + `ImageSearchResults` to `products/page.tsx`** Make three edits.
  (a) Add the store import directly below `import { searchApi } from "@/lib/api/search";` (line 14):
  ```ts
  import { useImageSearchStore } from "@/store/image-search-store";
  ```
  (b) In `ProductsListing`, add the mode flag next to `const useSearch = q.length >= 2;` (after line 51):
  ```ts
  const imageMode = sp.get("mode") === "image";
  ```
  and replace the two-way ternary in the returned JSX (currently lines 64–73, the `{useSearch ? (…) : (…)}` block) with this three-way branch:
  ```tsx
  {imageMode ? (
    <ImageSearchResults
      onClear={() => {
        useImageSearchStore.getState().reset();
        updateQuery({ mode: null });
      }}
    />
  ) : useSearch ? (
    <SearchResults q={q} onClear={() => updateQuery({ q: null })} />
  ) : (
    <ListingWithFilters
      categoryId={categoryId}
      brandId={brandId}
      sortKey={sortKey}
      updateQuery={updateQuery}
    />
  )}
  ```
  (c) Add the `ImageSearchResults` component. Insert it immediately after the `SearchResults` component's closing `}` (after line 242, before `interface SearchBodyProps`). It reads the store and reuses the existing `SearchResultCard` (defined at line 275) + `Spinner` + `EmptyState`:
  ```tsx
  function ImageSearchResults({ onClear }: Readonly<{ onClear: () => void }>) {
    const status = useImageSearchStore((s) => s.status);
    const results = useImageSearchStore((s) => s.results);
    const error = useImageSearchStore((s) => s.error);

    return (
      <>
        <div className="mb-8 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="font-display text-stone-900 text-4xl font-extrabold tracking-tight">
              Sản phẩm giống ảnh của bạn
            </h1>
            <p className="mt-2 text-sm text-stone-500">
              {status === "loading"
                ? "Đang tìm..."
                : `${results.length} sản phẩm`}
            </p>
          </div>
          <button
            type="button"
            onClick={onClear}
            className="text-primary-700 text-sm font-medium hover:underline"
          >
            Xoá tìm kiếm
          </button>
        </div>

        {status === "idle" && (
          <EmptyState
            title="Chưa có kết quả tìm bằng ảnh"
            description="Dùng nút 📷 trên thanh tìm kiếm để tìm sản phẩm bằng hình ảnh."
          />
        )}
        {status === "loading" && (
          <div className="flex justify-center py-20">
            <Spinner className="text-primary-700" size={32} />
          </div>
        )}
        {status === "error" && (
          <EmptyState
            title="Không tìm được sản phẩm từ ảnh"
            description={error ?? "Thử lại với ảnh khác nhé."}
          />
        )}
        {status === "success" && results.length === 0 && (
          <EmptyState
            title="Không tìm thấy sản phẩm giống ảnh"
            description="Thử chụp rõ hơn hoặc dùng ảnh khác."
          />
        )}
        {status === "success" && results.length > 0 && (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
            {results.map((p) => (
              <SearchResultCard key={p.id} product={p} />
            ))}
          </div>
        )}
      </>
    );
  }
  ```

- [ ] **Step 4: Type gate** Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && npx tsc --noEmit`  Expected: exits 0 with no output. Common failure to watch: unused `imageMode` (only if the ternary edit was skipped) or a missing store import.

- [ ] **Step 5: Manual verification** Start the app and drive the storefront path:
  - Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && npm run dev` (needs shop-api on :8080 and RAG on the `NEXT_PUBLIC_RAG_API_URL` port from `.env.local`). Open `http://localhost:3000`.
  - Click the new 📷 (camera) button in the header. Observe: the OS file picker opens and only offers JPG/PNG/WEBP.
  - Pick a product photo. Observe: the page navigates to `/products?mode=image`, the heading reads "Sản phẩm giống ảnh của bạn", and a centered spinner shows while the upload + hydration run.
  - Observe: the spinner is replaced by a grid of product cards (same card style as text search); each links to `/products/{slug}`.
  - Click "Xoá tìm kiếm". Observe: the URL loses `?mode=image` and the default all-products listing returns.
  - Empty case: point RAG at a catalog with no visual match (or a photo of something not stocked). Observe: "Không tìm thấy sản phẩm giống ảnh".
  - Error case: stop the RAG service, retry the 📷 upload. Observe: "Không tìm được sản phẩm từ ảnh. Thử lại nhé." (no stack trace, no infinite spinner).
  - Wrong-type case: rename a `.txt` to `.gif` and try to select it — the picker's `accept` blocks it; if forced, observe the "Định dạng ảnh không hỗ trợ" message.

- [ ] **Step 6: Commit**
  ```bash
  cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && \
  git add store/image-search-store.ts components/storefront/header-search.tsx "app/(storefront)/products/page.tsx" && \
  git commit -m "feat(storefront): image search via header camera button

  📷 button downscales the photo, calls searchByImage + listItemsByIds, stashes
  the hydrated ProductListItem[] in a zustand store, and navigates to
  /products?mode=image which renders it through the existing SearchResultCard
  grid (ProductGrid untouched). Loading/empty/error states covered.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 3: Entry 2 — chatbot image attach + paste → `ProductSourceCards`
**Files:**
- Modify `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui/components/chatbot/types.ts` (`ChatMessage` interface lines 3–17)
- Modify `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui/components/chatbot/chat-panel.tsx` (lucide import line 10; imports block lines 1–20; `ChatPanel` state lines 35–40; `send` ends line 122; `onKeyDown` lines 138–143; footer lines 218–244; `MessageBubble` return lines 303–324)

**Interfaces:**
- Consumes: `searchApi.searchByImage` (Task 1); `downscaleImage`, `makeThumbnailDataUrl` (Task 1); existing `ProductSourceCards` (renders `message.products` ids as cards — no change needed); existing `ChatMessage` shape.
- Produces: optional `image?: string` on `ChatMessage`; an image attach button + hidden input + paste handler + preview thumbnail in the chat footer; a `sendImageSearch(file, caption)` path that bypasses `sendChat` and emits an assistant message `{ content: "Đây là các sản phẩm giống ảnh của bạn", products: string[] }`.

- [ ] **Step 1: Add `image?` to `ChatMessage`** In `/home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui/components/chatbot/types.ts`, add a field to the `ChatMessage` interface immediately after the `placedOrderId?` field (after line 16):
  ```ts
    /** Client-only: small data-URL thumbnail of an image the user sent for
     *  visual search. Display-only; not sent to any backend. */
    image?: string;
  ```

- [ ] **Step 2: Add imports to `chat-panel.tsx`** Replace the lucide import at line 10 (`import { Send, Sparkles, Trash2, X } from "lucide-react";`) with:
  ```ts
  import { Image as ImageIcon, Send, Sparkles, Trash2, X } from "lucide-react";
  ```
  and add these two imports directly below `import { productsApi } from "@/lib/api/products";` (line 16):
  ```ts
  import { searchApi } from "@/lib/api/search";
  import { downscaleImage, makeThumbnailDataUrl } from "@/lib/image-downscale";
  ```

- [ ] **Step 3: Add attach state + ref** In `ChatPanel`, add beside the existing state (after line 39, `const inputRef = useRef<HTMLTextAreaElement>(null);`):
  ```ts
  const imageInputRef = useRef<HTMLInputElement>(null);
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [attachedPreview, setAttachedPreview] = useState<string | null>(null);
  ```

- [ ] **Step 4: Add attach + paste + image-send handlers** Insert immediately after the `send` `useCallback` closes (after line 122, before the `markPlaced` comment block):
  ```ts
  const attachImage = useCallback((file: File) => {
    const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];
    if (!ACCEPTED.includes(file.type)) return;
    setAttachedFile(file);
    setAttachedPreview(URL.createObjectURL(file));
  }, []);

  function onPaste(e: React.ClipboardEvent<HTMLTextAreaElement>) {
    const item = Array.from(e.clipboardData.items).find((it) =>
      it.type.startsWith("image/"),
    );
    const file = item?.getAsFile();
    if (file) {
      e.preventDefault();
      attachImage(file);
    }
  }

  const sendImageSearch = useCallback(
    async (file: File, caption: string) => {
      if (loading) return;
      // Small display thumbnail for the user bubble (upload uses a separate,
      // larger downscale below). LLM is bypassed — this hits /search/image.
      const thumb = await makeThumbnailDataUrl(file).catch(() => undefined);
      const userMsg: ChatMessage = {
        role: "user",
        content: caption.trim() || "Tìm sản phẩm bằng hình ảnh",
        image: thumb,
        ts: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setAttachedFile(null);
      setAttachedPreview(null);
      setLoading(true);
      setError(null);

      try {
        const downscaled = await downscaleImage(file);
        const { product_ids } = await searchApi.searchByImage(downscaled);
        // MessageBubble caps product cards at 4 — match it here (don't carry 4 unused ids).
        const ids = Array.from(new Set(product_ids)).slice(0, 4);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              ids.length > 0
                ? "Đây là các sản phẩm giống ảnh của bạn"
                : "Không tìm thấy sản phẩm giống ảnh.",
            products: ids,
            ts: Date.now(),
          },
        ]);
      } catch {
        setError("Không tìm được sản phẩm từ ảnh. Thử lại nhé.");
      } finally {
        setLoading(false);
      }
    },
    [loading],
  );
  ```

- [ ] **Step 5: Route Enter + Send button through the image path** Replace the `onKeyDown` body (lines 138–143) with:
  ```tsx
  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (attachedFile) sendImageSearch(attachedFile, input);
      else send(input);
    }
  }
  ```
  In the footer, replace the Send `<button>` `onClick` and `disabled` (lines 233–234) with:
  ```tsx
            onClick={() =>
              attachedFile ? sendImageSearch(attachedFile, input) : send(input)
            }
            disabled={loading || (!input.trim() && !attachedFile)}
  ```

- [ ] **Step 6: Add the attach button, hidden input, and preview to the footer** Add `onPaste={onPaste}` to the `<textarea>` (it currently spans lines 220–230; add the prop alongside `onKeyDown={onKeyDown}` at line 224). Then, inside the footer `<div className="flex items-end gap-2">` (opens line 219), add an attach button as the FIRST child (before the `<textarea>`):
  ```tsx
  <button
    type="button"
    onClick={() => imageInputRef.current?.click()}
    disabled={loading}
    aria-label="Đính kèm hình ảnh"
    className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg border border-stone-200 text-stone-500 transition-colors hover:bg-stone-100 hover:text-stone-700 disabled:opacity-40"
  >
    <ImageIcon size={16} />
  </button>
  <input
    ref={imageInputRef}
    type="file"
    accept="image/jpeg,image/png,image/webp"
    className="hidden"
    onChange={(e) => {
      const f = e.target.files?.[0];
      e.target.value = "";
      if (f) attachImage(f);
    }}
  />
  ```
  and add the preview block directly above that `<div className="flex items-end gap-2">` row (as the first child of the `<footer>`, before line 219):
  ```tsx
  {attachedPreview && (
    <div className="mb-2 flex items-center gap-2">
      <div className="h-14 w-14 overflow-hidden rounded-lg border border-stone-200">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={attachedPreview}
          alt="Ảnh đính kèm"
          className="h-full w-full object-cover"
        />
      </div>
      <button
        type="button"
        onClick={() => {
          setAttachedFile(null);
          setAttachedPreview(null);
        }}
        aria-label="Bỏ ảnh"
        className="rounded-md p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-700"
      >
        <X size={14} />
      </button>
    </div>
  )}
  ```

- [ ] **Step 7: Render the sent image in the user bubble** In `MessageBubble` (return at lines 303–324), add the image directly inside the outer flex-column `<div>`, immediately before the message-content bubble `<div>` (before line 305's `<div className={cn("max-w-[85%] …`):
  ```tsx
  {message.image && (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={message.image}
      alt="Ảnh đã gửi"
      className="max-w-[70%] rounded-2xl border border-stone-200"
    />
  )}
  ```
  (No change to `ProductSourceCards`: the assistant message sets `products` = the returned `product_ids`, and the existing `MessageBubble` logic at lines 296–302 already maps `message.products` → numeric ids → `<ProductSourceCards ids={…} />`.)

- [ ] **Step 8: Type gate** Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && npx tsc --noEmit`  Expected: exits 0 with no output. Watch for: `React.ClipboardEvent` / `KeyboardEvent` type mismatch (the file already imports `KeyboardEvent` as a type at lines 4–9 — reuse it; `ClipboardEvent` is referenced via the `React.` namespace which is available under `jsx: react-jsx`).

- [ ] **Step 9: Manual verification** With the app running (Task 2 Step 5 env), open the chatbot:
  - Click the bubble launcher, open the chat panel. Observe: a new image (🖼️) button sits left of the message textarea.
  - Click it, pick a product photo. Observe: a 56×56 preview thumbnail appears above the input with an ✕ to remove it. Click ✕ — the preview clears.
  - Re-attach, then press Enter (or click Send). Observe: a user bubble shows the thumbnail + caption "Tìm sản phẩm bằng hình ảnh", a loading indicator runs, then an assistant bubble reads "Đây là các sản phẩm giống ảnh của bạn" with a horizontally-scrollable row of product cards (each links to its product page).
  - Paste path: copy an image to the clipboard, click into the textarea, press Ctrl/Cmd+V. Observe: the preview thumbnail appears (no text is inserted into the textarea). Send it — same result as above.
  - Empty case: send an image with no visual match. Observe: "Không tìm thấy sản phẩm giống ảnh." and no cards.
  - Error case: stop RAG, send an image. Observe: the red error strip "Không tìm được sản phẩm từ ảnh. Thử lại nhé.".
  - Regression: send a TEXT message (no attachment). Observe: the normal `/chat` LLM reply still works unchanged.

- [ ] **Step 10: Commit**
  ```bash
  cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && \
  git add components/chatbot/types.ts components/chatbot/chat-panel.tsx && \
  git commit -m "feat(chatbot): image attach + paste → visual search cards

  Image button + onPaste + preview thumbnail in the chat input; sending with an
  image bypasses the LLM, POSTs to /search/image, and renders the returned
  product_ids through the existing ProductSourceCards with a fixed caption.
  ChatRequest to /chat is unchanged.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 4: Full-suite type gate + end-to-end manual verification
**Files:** none modified — verification only across the three touched files + two new files.

**Interfaces:** Consumes everything produced in Tasks 1–3. Produces: a signed-off type gate + E2E manual run.

- [ ] **Step 1: Clean type gate over the whole UI** Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && npx tsc --noEmit`  Expected: exits 0 with no output. (This is the repo's ONLY automated gate — there is no test runner.)

- [ ] **Step 2: Confirm no unrelated churn** Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && git diff --stat main..HEAD`  Expected: exactly these files appear — `lib/image-downscale.ts` (new), `lib/api/search.ts`, `store/image-search-store.ts` (new), `components/storefront/header-search.tsx`, `app/(storefront)/products/page.tsx`, `components/chatbot/types.ts`, `components/chatbot/chat-panel.tsx`. No changes to `ProductGrid`, `components/chatbot/api.ts`, or the `?q=` text-search code.

- [ ] **Step 3: E2E — storefront bar → grid** With shop-api (:8080), RAG (`NEXT_PUBLIC_RAG_API_URL`), and `npm run dev` running: upload a product photo via the header 📷. Observe end-to-end: picker → `/products?mode=image` → spinner → a grid of visually-similar product cards. Confirm the ORDER of cards matches RAG's ranking as hydrated by `list-items` (visible-only). Empty and error variants behave as in Task 2 Step 5.

- [ ] **Step 4: E2E — chat paste → cards** In the chatbot, paste an image (Ctrl/Cmd+V in the textarea) and send. Observe end-to-end: user bubble with thumbnail → loading → assistant "Đây là các sản phẩm giống ảnh của bạn" + a card row. Confirm a plain text chat still returns a normal LLM answer (ChatRequest unchanged).

- [ ] **Step 5: Contract cross-check** Confirm both entry points read the SAME contract: in `lib/api/search.ts` the response type is `{ product_ids: string[] }` (H1) and the upload field is `fd.append("file", …)` (H2); the chatbot reuses `searchApi.searchByImage` (same function), so there is a single source of truth. Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && grep -n "fd.append(\"file\"" lib/api/search.ts && grep -n "product_ids" lib/api/search.ts`  Expected: the `file` append prints once; `product_ids` appears in the `ImageSearchResponse` type and the `searchByImage` return.

- [ ] **Step 6: No commit** This task changes no files; nothing to commit. If Steps 1–2 revealed drift, fix in the owning task and re-run.
