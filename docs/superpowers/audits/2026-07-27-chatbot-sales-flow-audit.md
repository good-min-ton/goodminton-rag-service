# Báo cáo chẩn đoán root-cause — Goodminton Sales Chatbot

## Tóm tắt điều hành (Executive Summary)

Nguyên nhân cấu trúc đứng sau gần như toàn bộ triệu chứng là: **endpoint `/chat` hoàn toàn stateless và "câm ngữ cảnh" (context-blind)** — mỗi lượt chỉ embed đúng câu mới nhất rồi chạy **một** truy vấn cosine top-k=5 không có `WHERE` trên toàn bộ `kb_chunks`, không có intent, không có category, không có server-side state, không có order state machine. Product cards không được sinh cùng câu trả lời mà bị **scrape ngược từ text** bằng so khớp substring.

Phân tách lỗi theo layer (đây KHÔNG phải bài toán sửa prompt):
- **Retrieval pipeline (architectural):** P1, P2 — thiếu intent/category extraction, thiếu category filter, thiếu query contextualization.
- **Backend orchestration (architectural):** P2, P3, P4, P5 — thiếu session/order state, thiếu sanitize tool-result, thiếu contract `display_products` gắn theo message.
- **Frontend state (architectural, nhỏ hơn):** P3, P5 — render card không phụ thuộc order state, fallback về raw sources.
- **Prompt (chỉ phụ trợ):** P3, P4 — prompt không có khái niệm phase và không cấm echo JSON, nhưng đây chỉ là lớp phụ; sửa prompt đơn thuần **không** đóng được bất kỳ P nào.

---

## Cơ chế hiện tại (pipeline thực tế đang chạy)

- **Stateless mỗi request.** `/chat` không lưu state phía server. `session_id` (`schemas.py:14`) chỉ dùng làm trace tag cho Langfuse tại `chat.py:38`, không key vào bất kỳ store nào. State cross-turn duy nhất là `chat_history` do frontend replay từ localStorage (tối đa 20 lượt).
- **Retrieval chỉ nhìn câu mới nhất.** `query = request.message.strip()` (`chat.py:33`) → `embed(query)` (`chat.py:49`) → `search(query_vec)` (`chat.py:54`). `chat_history` chỉ được append vào message list của LLM tại `chat.py:79-81`, **sau khi** retrieval đã chạy xong.
- **Một truy vấn cosine không filter.** `RetrievalService.search` (`retrieval.py:30-40`) là `ORDER BY embedding <=> $1 LIMIT $2`, không `WHERE` doc_type/category, top_k=5 (`config.py:25`).
- **Category không phải cột query được.** Category chỉ là free text nhúng vào `content` lúc index (`indexer.py:43-49`, dòng `"Danh mục: ..."`). `kb_chunks` có cột `metadata JSONB` nhưng indexer không ghi và retrieval không đọc (`indexer.py:84-93`).
- **Product cards scrape từ text.** `_extract_recommended` (`chat.py:278-306`) build map name-core → product_id từ chunk + kết quả `recommend_similar_products`, rồi chỉ giữ id nào có name-core xuất hiện dạng substring trong answer (`chat.py:300-304`).
- **Không có order state machine.** "Order draft" chỉ là kết quả `prepare_order` cuối cùng của **chính lượt đó** (`chat.py:154-157`), reset `None` mỗi lượt (`chat.py:111`). Đặt hàng thực hiện hoàn toàn ở frontend (`OrderConfirmCard` → `ordersApi.create`), backend/LLM không bao giờ biết đơn đã đặt.

---

## P1 — Recommendations không khớp intent

**(a) Triệu chứng:** hỏi quần → ra áo; hỏi quần + giày → chỉ ra quần; gợi ý danh mục không liên quan.

**(b) Root cause:**
- Không có intent classification ở bất kỳ đâu; handler chạy thẳng message → embed → search, không có bước trung gian (`chat.py:33-54`); và retrieval không có tham số nào để một intent đã phân loại có thể ràng buộc kết quả (`retrieval.py:23-51`). Retrieval chạy **trước** LLM/tool loop (dòng 54 vs 83) nên tool-calling không thể thu hẹp ngữ cảnh về sau.
- Vector search chạy trên toàn bảng, không filter category/doc_type (`retrieval.py:30-40`); category chỉ là text nhúng lúc index (`indexer.py:43-49`) → không ràng buộc được lúc query, thậm chí có thể trả về doc help phi-product.
- Multi-category không được honor: cả câu multi-intent bị embed thành **một** vector trộn (`chat.py:49`), không decompose theo category; một global top-k, không MMR/quota/merge (`retrieval.py:30-40`); top_k=5 (`config.py:24-25`) lại còn bị dedupe theo `source_id` (`chat.py:309-320`) nên 5 slot có thể chỉ là 1-2 sản phẩm.
- Ranking thuần cosine, không grouping/diversity/quota (`retrieval.py:26-40`) → chunk lệch category gần hơn một chút là chiếm slot.
- `recommend_similar_products` xếp hạng bằng cosine centroid không ràng buộc, `WHERE` chỉ có `doc_type='product'` và `source_id<>$1`, không guard category (`similar.py:44-57`, `tools.py:60-75`) → cross-category by mechanism.
- Card cuối cùng cũng chỉ là substring-scrape từ answer trên pool không filter (`chat.py:278-306`), chỉ propagate được lỗi upstream, không sửa được.

**(c) Layer:** retrieval pipeline (chính) + backend orchestration (card selection).

**(d) Missing vs wrong:** chủ yếu **code is missing** (intent + category filter + query decomposition không tồn tại); phần ranking/card-scrape là **code tồn tại nhưng không có category guard**.

---

## P2 — Không có conversation context

**(a) Triệu chứng:** "Mua quần?" rồi "Rẻ nhất?" → bot không biết "rẻ nhất" là quần; search cả catalog hoặc gợi ý áo.

**(b) Root cause:**
- Retrieval chỉ build từ câu mới nhất; `chat_history` không bao giờ vào vector query (`chat.py:33-81`, đặc biệt 49/54 vs 79-81). "Rẻ nhất?" bị embed đơn độc, k-NN trả về thứ gần chữ "cheapest" nhất — không có scope "quần", cũng không có logic giá thật.
- Không có server-side session state (`chat.py:37-42`): không có `current_product_category`, `intent`, `price_preference`, `selected_product_id`, `order_status` ở đâu cả.
- Wire contract không có field state: `ChatRequest`/`ChatResponse` chỉ có message + chat_history vào, answer/sources/products/order_draft ra (`schemas.py:11-47`) → không có kênh để trả state ra và echo lại lượt sau.
- Không có bước query-rewrite/condense để giải elliptical ("rẻ nhất" → "quần rẻ nhất") trước khi embed (`chat.py:43-54`).
- Tool loop cũng không cứu được: cả 4 tool đều seed bằng ID, không nhận free text nên không thể re-issue truy vấn semantic theo history (`tools.py`), và ID hợp lệ chỉ đến từ retrieval câm ngữ cảnh đó.
- Frontend replay history từ localStorage mỗi lượt, chỉ map `{role, content}`, không gửi `session_id` (`chat-panel.tsx:102-110`; type tại `components/chatbot/types.ts:27-30`).

**(c) Layer:** retrieval pipeline + backend orchestration + frontend.

**(d) Missing vs wrong:** **code is missing** (session state, query contextualization, state fields trong schema). Phần "history vào prompt" thì **tồn tại nhưng lệch chỗ** — nó nuôi LLM, không nuôi retrieval.

---

## P3 — Order flow không được quản lý

**(a) Triệu chứng:** chọn quần xong bot vẫn gợi áo/giày; reco vẫn hiện sau prompt confirm; hội thoại loop, không đóng đơn; bot quên sản phẩm đã chọn.

**(b) Root cause:**
- **Không tồn tại order/conversation state machine.** Máy trạng thái kỳ vọng `BROWSING→PRODUCT_SELECTED→CHECKING_STOCK→WAITING_CONFIRMATION→ORDER_CONFIRMED→COMPLETED` không có ở đâu; `ChatResponse` không có `conversation_state`/`order_status` (`schemas.py:41-47`); order_draft chỉ suy ra từ lượt hiện tại (`chat.py:154-157`, reset tại 111). Đây là root cấu trúc của mọi triệu chứng P3.
- Reco không bị suppress ở backend: retrieval + `_extract_recommended` chạy vô điều kiện mỗi lượt, `return ChatResponse` set cả `products` lẫn `order_draft` không loại trừ nhau (`chat.py:86-94`). Trên lượt confirm, answer nhắc tên sản phẩm đang đặt → card sản phẩm đó ride kèm card confirm.
- Prompt định nghĩa quy trình tuyến tính, không phải state machine, và luật tư vấn số 4 luôn bảo "gợi ý 2-3 sản phẩm" mà không có điều kiện tắt sau khi có draft (`prompts.py:16-26`). System prompt chỉ được bổ sung danh sách product_id hợp lệ (`chat.py:64-76`), không có chỉ thị "đang chờ confirm, ngừng gợi ý".
- Đặt hàng chỉ ở frontend, không feed lại backend/LLM (`chat-panel.tsx:106-109` map chỉ `{role, content}`; backend rebuild message từ chat_history tại `chat.py:78-81`; placement là REST riêng tới shop-api `order-confirm-card.tsx:67-89`) → LLM không bao giờ nhận tín hiệu "đã đặt", có thể tiếp tục gợi ý/re-draft.
- Sản phẩm/variant đã chọn không persist: `ChatRequest` không có field inbound cho `selected_product_id`/`variant_id`/`order_draft` (`schemas.py:11-14`), `ChatMessage` chỉ `{role, content}` (`schemas.py:6-8`) → LLM phải suy lại từ text mỗi lượt.
- Frontend render card ngay trên lượt confirm: `productIds` tính cho mọi message non-user, không biết order state, và ProductSourceCards render trên guard độc lập với OrderConfirmCard (`chat-panel.tsx:411-451`, guard tại 444/445), fallback về top-3 retrieved sources khi `products` rỗng.

**(c) Layer:** backend orchestration (chính) + frontend + prompt (phụ).

**(d) Missing vs wrong:** state machine, order-feedback, selected-product persistence là **code is missing**; suppression logic ở backend là **missing**; prompt/frontend render là **code tồn tại nhưng thiếu điều kiện state**.

---

## P4 — Tool results hiện raw cho user

**(a) Triệu chứng:** answer trả về JSON thô như `{"product_id":164,"size":"M","quantity":1}`.

**(b) Root cause:**
- Phát hiện tool-call chỉ dựa vào field structured `message.tool_calls` của Ollama (`chat.py:122`; `llm.py:47-61` POST raw JSON, không parse). Model nhỏ (3B) thường không điền field này mà phát ra call dưới dạng JSON text trong `content` → `tool_calls` rỗng, guard `chat.py:130` kích hoạt, `chat.py:131` trả `content` **nguyên văn**. Không có fallback parser cho tool-call phát ra dạng content.
- Answer không bao giờ được sanitize trước khi vào `ChatResponse.answer` (`chat.py:83-94`); `_extract_recommended` trả answer y nguyên (`chat.py:306`); `ChatResponse.answer` là `str` trần không validator (`schemas.py:42`). Backend parse JSON thành `order_draft`/`products` đúng, nhưng không có guard tương đương cho text.
- Prompt không có luật cấm echo JSON tool-call/tool-result, cũng không có luật "chỉ trả lời bằng ngôn ngữ tự nhiên" (`prompts.py:22-31`); mọi tool observation lại được feed lại dưới dạng JSON string (`tools.py:167`, append tại `chat.py:158`).
- Nhánh fallback khi model stuck cũng không an toàn: `llm.chat(messages)` chạy trên message list vẫn đầy JSON tool-result, trả về nguyên văn không sanitize (`chat.py:160-182`).

**(c) Layer:** backend orchestration (chính) + prompt (phụ).

**(d) Missing vs wrong:** cơ chế phát hiện native field **tồn tại và đúng**; cái **thiếu** là (1) fallback parser cho tool-call-as-content, (2) bước sanitize/strip JSON khỏi answer, (3) luật anti-echo trong prompt.

---

## P5 — Text và product list lệch nhau

**(a) Triệu chứng:** text nói về quần nhưng card bên dưới là áo.

**(b) Root cause:**
- Card là **scrape hậu kỳ** từ prose, không co-generate: LLM viết answer trong tool loop và bị prompt cấm nhắc ID (`chat.py:67-68`), nên `_extract_recommended` phải substring-match name-core vào `answer.lower()` (`chat.py:297-306`). Gate này áp cả lên output structured của `recommend_similar_products` — sản phẩm có product_id thật vẫn bị bỏ nếu core không xuất hiện trong prose. Fragility hai chiều: (a) paraphrase/lệch dấu/khoảng trắng → `find()` miss, drop sản phẩm dù text vẫn recommend (core lowercase nhưng không normalize dấu/space); (b) core ngắn substring-collide, và vì `core_to_id` build từ mọi product chunk không so category → sản phẩm sai category bị card (P1). → xem P1 để rõ gốc category-blind.
- Fallback frontend: khi `products` rỗng, UI hiện 3 raw retrieval sources (`chat-panel.tsx:408-423`), ProductSourceCards chỉ filter `isVisible` (`chat-panel.tsx:458-469`), không match category, không cross-check với prose → card có thể là sản phẩm text không nhắc.
- Contract thiếu field: `ChatResponse` không có `intent`/`categories`/`conversation_state`/`display_products` (`schemas.py:41-47`); `products` chỉ là `list[str]` id. Marker `<<products: ...>>` chỉ tồn tại trong comment (stale/gây hiểu nhầm), nguồn thật là substring scrape.
- Card không được clear khi vào order flow: guard render độc lập với `order_draft` (`chat-panel.tsx:444-451`); backend luôn emit `sources` từ retrieval mỗi lượt kể cả lượt order (`chat.py:89-94`, `_unique_sources` giữ doc_type='product' tại 192-201).

**(c) Layer:** backend orchestration (scrape/contract) + frontend (fallback/clear).

**(d) Missing vs wrong:** contract `display_products` là **missing**; scrape và fallback là **code tồn tại nhưng sai thiết kế** (binding suy sau, decoupled khỏi text).

> **Lưu ý trung thực (triệu chứng nghi ngờ KHÔNG tái hiện):** Nghi vấn ban đầu của P5 rằng "frontend giữ lại recommendations của lượt trước" **không đúng với code**. Auditor xác nhận binding per-message là **chính xác** — không có cross-turn carryover (`chat-panel.tsx:111-121`, `mechanism_status: present_ok`): mỗi assistant turn append một object bất biến mang `res.products/res.sources` riêng; mutation duy nhất `markPlaced` (202-210) chỉ ghi `placedOrderId`, không đụng products/sources. Vì vậy P5 phải đến từ **logic trong-lượt** (fallback `fromSources.slice(0,3)` tại 417-418 + scrape backend), không phải state cũ. Danh sách `rejected_or_uncertain` rỗng; ngoài điểm này không có triệu chứng nào bị bác bỏ.

---

## Lộ trình sửa ưu tiên (theo leverage)

Sắp theo đòn bẩy: fix đóng nhiều P trước. Ký hiệu quy mô: **S** (nhỏ, <1 ngày), **M** (vừa, 1-3 ngày), **L** (lớn, >3 ngày, cần thiết kế).

| # | Đóng P | Layer | Xây gì | Quy mô | Khớp đề xuất user? |
|---|--------|-------|--------|--------|---------------------|
| 1 | P2, P3, P5 | Backend orchestration | **Server-side conversation/session state** keyed theo `session_id`: lưu `{intent, categories, price_preference, selected_product_id, order_status}` (store/cache). Thêm field state vào `ChatRequest`/`ChatResponse` (`schemas.py:11-47`), frontend gửi `session_id` (`chat-panel.tsx:102-110`, `types.ts:27-30`). | L | **Có** — khớp chính xác state `{intent, categories, price_preference, selected_product_id, order_status}` user đề xuất |
| 2 | P1, P2 | Retrieval pipeline | **Intent/category extraction + category-filtered retrieval + query contextualization.** Thêm cột/dùng `metadata JSONB` cho category (ghi tại `indexer.py:84-93`), thêm `WHERE` category/doc_type + per-category retrieve-then-merge/quota vào `search()` (`retrieval.py:30-40`), thêm bước condense-question dùng history trước `embed()` (`chat.py:43-54`), decompose multi-intent thành sub-query. Thêm guard category vào `similar.py:44-57`. | L | Một phần — user chỉ rõ hành vi kỳ vọng (chỉ quần / mỗi category / no cross-category), khớp nhưng user chưa đặc tả cơ chế retrieval |
| 3 | P1, P5 | Backend + Frontend | **Contract `display_products` gắn theo đúng message.** `ChatResponse` trả `{message, intent, categories, display_products (object đã resolve), conversation_state}` thay vì scrape substring (`chat.py:278-306`); frontend render **chỉ** `display_products` của message đó, bỏ fallback raw sources (`chat-panel.tsx:408-423`). | M | **Có** — khớp chính xác contract `{message, intent, categories, display_products, conversation_state}` user đề xuất |
| 4 | P3 | Backend + Prompt + Frontend | **Order state machine** `BROWSING→…→COMPLETED`. Trong `WAITING_CONFIRMATION`: KHÔNG gọi reco RAG / không show sản phẩm mới, chỉ cho confirm/cancel/đổi qty-size; sau confirm ngừng gợi ý tới buy request mới. Suppress `products` khi có `order_draft` (`chat.py:86-94`), gate render card theo state (`chat-panel.tsx:444-451`), feed order-confirmed về backend. Prompt bổ sung điều kiện tắt luật gợi ý theo phase (`prompts.py:16-26`). | M | **Có** — khớp chính xác state machine user đề xuất. **Giữ nguyên** `prepare_order` read-only (chỉ price+stock, `tools.py:196-275`) |
| 5 | P4 | Backend + Prompt | **Sanitize tool-result khỏi reply.** Thêm fallback parser cho tool-call phát dạng content-JSON (`chat.py:122-131`), strip/validate JSON khỏi `answer` trước `ChatResponse` (`chat.py:83-94`, `schemas.py:42`), áp cả nhánh stuck-fallback (`chat.py:160-182`); thêm luật anti-echo "chỉ ngôn ngữ tự nhiên" vào prompt (`prompts.py:22-31`). | S–M | **Có** — khớp kỳ vọng "tool call/result không bao giờ vào reply; backend parse JSON trước; LLM chỉ nhận kết quả normalized" |

Ghi chú leverage: Fix #1 và #2 là hai trục architectural gánh phần lớn triệu chứng (P1+P2+P3+P5). Sửa prompt (phần của #4, #5) là **bổ trợ**, tự thân không đóng P nào — đúng như yêu cầu của user rằng đây không phải bài toán prompt-only.

---

## Rủi ro / lưu ý khi sửa

- **Model 3B (GPU 6GB):** năng lực tool-calling và tuân prompt yếu — đây chính là gốc P4. Đừng dựa vào prompt để ép model không echo JSON; phải có sanitize phía backend (#5). Query-rewrite/intent extraction (#2) nếu giao cho 3B sẽ kém tin cậy — cân nhắc rule-based/regex cho category thông dụng hoặc model lớn hơn (memory ghi live cần 14B cho order-assistant).
- **Giữ `prepare_order` read-only:** `_prepare_order` (`tools.py:196-275`) hiện chỉ price + stock-check, không tạo/track đơn; đặt hàng do frontend (`OrderConfirmCard` → `ordersApi.create`). State machine (#4) phải giữ nguyên thiết kế này — chỉ thêm tín hiệu ORDER_CONFIRMED feed về backend, KHÔNG để backend/LLM tự tạo đơn.
- **Nhánh image-search chưa merge:** đang ở `feat/image-search`, path SigLIP (`image_search.py`/`image_indexer.py`) tách khỏi path text (`embedding_svc.embed` tại `chat.py:49`). Đừng để refactor retrieval (#2) va chạm image path; category filter phải áp đúng path text.
- **`session_id` hiện chỉ là trace tag** (`chat.py:38`): tái dùng làm key store (#1) cần đảm bảo frontend gửi nó (hiện KHÔNG gửi) và xử lý session mới/thiếu id.
- **Binding per-message đang đúng — đừng phá:** `chat-panel.tsx:111-121` đã bind card đúng theo từng message. Khi thêm `display_products` (#3), giữ nguyên tính bất biến này; chỉ bỏ fallback raw sources, không đụng cơ chế append/markPlaced.
- **Chunk dài mất tín hiệu category:** với `chunk_size=500/overlap=50` (`config.py:56`), sản phẩm text >500 ký tự sẽ mất dòng "Danh mục" ở chunk sau — nên category phải thành cột/metadata thật (#2), không dựa vào in-embedding text.
---

## Addendum — đối chiếu với `origin/main` mới nhất (sau PR #2)

Audit gốc chạy trên checkout `feat/image-search` (fork từ local `main` cũ), **thiếu 2 commit** đã có trên `origin/main`:
- `fcf243e fix(chat): pair product ids with names in tool-calling hints`
- `f331dc9 Merge PR #2 (fix/chat-product-id-grounding)`

**PR #2 chỉ đụng 2 file:** `app/routers/chat.py` + `app/core/prompts.py`. Sáu file pipeline còn lại (`retrieval.py`, `tools.py`, `llm.py`, `schemas.py`, `similar.py`, `indexer.py`) **identical** với bản audit.

**PR #2 làm gì:** thay `_extract_product_ids` (list id trơ) → `_extract_product_catalog` (cặp `id: tên`); system prompt liệt kê `- pid: name` để model chọn id khớp TÊN khách hỏi; thêm 2 luật prompt: (1) chọn đúng mẫu khi nhiều tên gần giống, (2) không chắc thì DỪNG gọi tool, hỏi lại 2-3 tên.

**Ảnh hưởng lên 5 root cause — đối chiếu lại trên origin/main (line numbers cập nhật):**
- **P1:** root cause GIỮ NGUYÊN. Retrieval vẫn category-blind (`retrieval.py:30-40` identical). PR #2 chỉ giúp chọn đúng **variant trong số candidate đã retrieve** (giảm nhẹ sub-case gọi nhầm mẫu + hành vi hỏi-lại), KHÔNG lọc category. Chi tiết "system prompt là danh sách id trơ" nay đổi thành cặp `id: tên` (`chat.py:62-72`) — vẫn không có chỉ thị phase/state.
- **P2:** GIỮ NGUYÊN — `query=request.message` (`chat.py:33`), `embed` (49), `search` (54), history chỉ nuôi LLM (82). Không có state.
- **P3:** GIỮ NGUYÊN — `ChatResponse` set cả products lẫn order_draft (`chat.py:92-96`), không suppress; không state machine (`schemas.py` identical).
- **P4:** GIỮ NGUYÊN — guard `if not tool_calls: return msg.get("content")` (`chat.py:133-134`) không nằm trong diff PR #2; `llm.py` identical; không sanitize.
- **P5:** GIỮ NGUYÊN — card scrape `_extract_recommended` (`chat.py:281-309`, `answer.lower()` 300, `.find(core)` 304) không đổi; contract vẫn thiếu `display_products`.

**Line-shift:** chỉ citation trong `chat.py` (+~3) và `prompts.py` (+2) lệch nhẹ so với báo cáo gốc; 6 file kia chính xác. Lộ trình sửa #1–#5 **không đổi**.

**Lưu ý cho việc sửa:** local `main` đang behind `origin/main` 2 commit → khi bắt đầu fix, **branch từ `origin/main`** (đã có PR #2), không từ `feat/image-search`.

### Addendum 2 — shop-ui `main` cũng đã tiến (2 commit)

Audit frontend dựa trên `b90b39e`; `origin/main` shop-ui giờ là `dae04d2`, thêm:
- `559bbef feat(chatbot): send stable session_id with each /chat call` — thêm `components/chatbot/session.ts` (`getChatSessionId()` per-browser, localStorage `gm.chat-session-id`; có `resetChatSessionId()`), `ChatRequest.session_id?` (`types.ts`), và `sendChat({... session_id})` (`chat-panel.tsx`).
- `dae04d2 fix(chatbot): strict guest gate + 401 fallback on order confirm` — `order-confirm-card.tsx` (auth gating; không đụng root cause P3 state-machine).

**Ảnh hưởng audit:**
- **P2:** chi tiết "frontend không gửi session_id" nay **SAI** — frontend ĐÃ gửi. NHƯNG backend RAG (`origin/main`) vẫn chỉ dùng `session_id` cho `propagate_attributes` tracing (`chat.py:38`), **chưa key state** → root cause "không có server-side session state" GIỮ NGUYÊN.
- **Fix #1 giảm việc:** dây wire session_id (frontend→backend) đã có sẵn; #1 còn lại = backend store state theo `session_id` + thêm state fields vào `ChatResponse`. **Lưu ý thiết kế:** session_id hiện per-browser-vĩnh-viễn (không phải per-conversation) → cần chốt: "conversation" = browser session hay reset khi clear chat (đã có `resetChatSessionId()` nhưng cần kiểm tra đã wire vào nút clear chưa).
