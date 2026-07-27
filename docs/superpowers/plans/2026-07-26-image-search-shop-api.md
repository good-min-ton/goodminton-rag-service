# Image Search — shop-api Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax.

**Goal:** Give the RAG image-search subsystem the shop-api pieces it needs: a `product_image_embeddings` table (schema owner), an internal endpoint that lists a product's image URLs for indexing, a public order-preserving `list-items` hydration endpoint, and an image-changed event so RAG can re-index on upload/delete.

**Architecture:** shop-api owns the shared `goodminton` Postgres schema. A new Flyway migration adds `product_image_embeddings` (written by RAG, never by JPA — no shop-api entity). RAG reads image URLs via `GET /api/internal/products/{id}/images` (X-Internal-Key) and hydrates search hits via public `GET /api/products/list-items?ids=`. `uploadProductImage`/`deleteProductImage` publish `ProductChangedEvent.updated(productId, Set.of("images"))` over the existing `product.*` RabbitMQ binding so the RAG consumer re-indexes.

**Tech Stack:** Spring Boot 3 (Java 21), Spring Data JPA, Flyway, pgvector (`pgvector/pgvector:pg15`), Spring AMQP (RabbitMQ), JUnit 5 + Mockito + AssertJ + MockMvc.

## Global Constraints
- **Shared DB:** one Postgres database `goodminton` (`pgvector/pgvector:pg15`); shop-api owns the schema. Do NOT add a JPA entity for `product_image_embeddings` — RAG writes it; shop-api only ships the DDL.
- **Vector dimension:** the new `product_image_embeddings.embedding` is `vector(768)` (SigLIP `google/siglip-base-patch16-224`). This is SEPARATE from `kb_chunks.embedding vector(1024)` (bge-m3) created in V7. Do not touch `kb_chunks`.
- **Migration V-number:** next is `V10` (latest present is `V9__add_payos_payment_method.sql`). `CREATE EXTENSION IF NOT EXISTS vector` already ran in V7 — reference it, do not re-create.
- **FK policy:** `product_id` FK → `products(id) ON DELETE CASCADE` (cleanup on product delete). NO `resource_id` FK (avoids write-time races between RAG indexing and shop-api resource writes — review #11).
- **HNSW index:** `USING hnsw (embedding vector_cosine_ops)` — mirror V7's cosine HNSW pattern.
- **H8 — is_visible + order:** `GET /api/products/list-items?ids=` MUST filter `is_visible` server-side (reuse `ProductRepository.findVisibleByIdInWithVariants`) and RETURN results in the REQUESTED id order, dropping ids that are hidden/missing. Batch the thumbnail lookup (no N+1).
- **X-Internal-Key:** `GET /api/internal/products/{id}/images` is guarded by `InternalAuthFilter` (`X-Internal-Key` on every `/api/internal/**`). Internal endpoints return RAW DTOs (no `ApiResponse` wrapper) — matches `InternalProductController.getForRag`. RAG's `ImageIndexer` consumes fields `resourceId`, `url`, `sortOrder`.
- **H1 contract (hydration half):** RAG `/search/image` returns `{"product_ids": ["42","17",…]}` (strings, ranked); shop-api's `list-items` is the hydration half that turns those ids into visible, ordered `ProductListItemResponse`s.
- **Ship the event with the RAG consumer half (decision #4):** publish `ProductChangedEvent.updated(productId, Set.of("images"))` from BOTH `uploadProductImage` and `deleteProductImage` (they currently publish nothing). The marker string **`"images"`** is the cross-subsystem contract the RAG `product_consumer` keys on. Routed via the existing `product.*` binding, published AFTER_COMMIT only (existing `ProductEventPublisher`).
- **Verification:** commands use `./mvnw test` (needs local infra: Postgres/Redis/RabbitMQ per `docker-compose.infra.yml`). CI = maven **Build & Test** running `mvn -B verify` with pg/redis/rabbitmq service containers. A sandbox JDK caveat exists but CI handles it — the plan's commands are the normal `./mvnw test`.

---

### Task 1: Flyway migration `V10__add_product_image_embeddings.sql`
**Files:**
- Create: `src/main/resources/db/migration/V10__add_product_image_embeddings.sql`
- Test (create): `src/test/java/com/lezh1n/goodminton_shop_api/repositories/ProductImageEmbeddingsMigrationTest.java`

**Interfaces:**
- Consumes: existing `products(id)`; `vector` extension (V7).
- Produces: table `product_image_embeddings(product_id int, resource_id int PK, url text, embedding vector(768))`, HNSW cosine index `idx_pie_embedding`, btree `idx_pie_product`, FK `fk_pie_product → products(id) ON DELETE CASCADE`.

- [ ] **Step 1: Write the failing test** Create `ProductImageEmbeddingsMigrationTest.java`. It persists a bare visible product (satisfies the FK), inserts two 768-dim embedding rows via native SQL, asserts cosine ordering, and asserts `ON DELETE CASCADE`. (This is a `@DataJpaTest` against real Postgres with Flyway run — same slice as `ProductRepositoryHybridTest`.)
```java
package com.lezh1n.goodminton_shop_api.repositories;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;

import com.lezh1n.goodminton_shop_api.entities.Brand;
import com.lezh1n.goodminton_shop_api.entities.Category;
import com.lezh1n.goodminton_shop_api.entities.Product;

import jakarta.persistence.EntityManager;

@DataJpaTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
class ProductImageEmbeddingsMigrationTest {

    @Autowired
    TestEntityManager em;

    private Integer productId;

    // Build a 768-dim pgvector literal with a single "hot" 1.0 component (rest 0.0).
    private static String vec(int hot) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < 768; i++) {
            if (i > 0) sb.append(',');
            sb.append(i == hot ? "1" : "0");
        }
        return sb.append(']').toString();
    }

    private void insertEmbedding(int resourceId, String embedding) {
        EntityManager delegate = em.getEntityManager();
        delegate.createNativeQuery(
                "INSERT INTO product_image_embeddings(product_id, resource_id, url, embedding) "
                        + "VALUES (:pid, :rid, :url, CAST(:emb AS vector))")
                .setParameter("pid", productId)
                .setParameter("rid", resourceId)
                .setParameter("url", "http://img/" + resourceId)
                .setParameter("emb", embedding)
                .executeUpdate();
    }

    @BeforeEach
    void setUp() {
        LocalDateTime now = LocalDateTime.now();
        Category category = em.persist(Category.builder().name("Rackets").build());
        Brand brand = em.persist(Brand.builder().name("Yonex").build());
        Product product = Product.builder()
                .category(category).brand(brand)
                .name("p").slug("p").isVisible(true)
                .createdAt(now).updatedAt(now)
                .variants(new ArrayList<>())
                .build();
        productId = em.persistAndFlush(product).getId();
    }

    @Test
    void cosineOrdering_returnsNearestFirst() {
        insertEmbedding(1, vec(0)); // aligned with the query vector -> distance 0
        insertEmbedding(2, vec(5)); // orthogonal -> distance 1
        em.flush();

        @SuppressWarnings("unchecked")
        List<Integer> ordered = em.getEntityManager().createNativeQuery(
                "SELECT resource_id FROM product_image_embeddings "
                        + "ORDER BY embedding <=> CAST(:q AS vector) ASC")
                .setParameter("q", vec(0))
                .getResultList();

        assertThat(ordered).containsExactly(1, 2);
    }

    @Test
    void deletingProduct_cascadeDeletesEmbeddings() {
        insertEmbedding(3, vec(0));
        em.flush();

        em.getEntityManager().createNativeQuery("DELETE FROM products WHERE id = :pid")
                .setParameter("pid", productId)
                .executeUpdate();
        em.flush();

        Number remaining = (Number) em.getEntityManager().createNativeQuery(
                "SELECT COUNT(*) FROM product_image_embeddings WHERE product_id = :pid")
                .setParameter("pid", productId)
                .getSingleResult();

        assertThat(remaining.longValue()).isZero();
    }
}
```
- [ ] **Step 2: Run test to verify it fails** Run: `./mvnw test -Dtest=ProductImageEmbeddingsMigrationTest`  Expected: FAIL — Flyway/JPA error `relation "product_image_embeddings" does not exist` (migration not yet created).
- [ ] **Step 3: Write minimal implementation** Create `V10__add_product_image_embeddings.sql`:
```sql
-- ============================================================
-- V10__add_product_image_embeddings.sql
-- Image-search vector store (SigLIP, 768-dim) — SEPARATE from kb_chunks (bge-m3, 1024).
-- Rows are written by RAG's ImageIndexer, never by shop-api JPA (no entity).
-- Extension `vector` already created in V7.
-- product_id FK -> products(id) ON DELETE CASCADE (cleanup on product delete).
-- NO resource_id FK: avoids write-time races between RAG indexing and resource writes.
-- ============================================================
CREATE TABLE
    product_image_embeddings (
        product_id INTEGER NOT NULL,
        resource_id INTEGER PRIMARY KEY,
        url TEXT NOT NULL,
        embedding VECTOR (768) NOT NULL, -- google/siglip-base-patch16-224
        CONSTRAINT fk_pie_product FOREIGN KEY (product_id)
            REFERENCES products (id) ON DELETE CASCADE
    );

-- HNSW cosine index (mirrors V7's kb_chunks pattern): fast query, no training.
CREATE INDEX idx_pie_embedding ON product_image_embeddings USING hnsw (embedding vector_cosine_ops);

-- Lookup / cascade support for all embeddings of one product.
CREATE INDEX idx_pie_product ON product_image_embeddings (product_id);
```
- [ ] **Step 4: Run test to verify it passes** Run: `./mvnw test -Dtest=ProductImageEmbeddingsMigrationTest`  Expected: PASS (both `cosineOrdering_returnsNearestFirst` and `deletingProduct_cascadeDeletesEmbeddings`).
- [ ] **Step 5: Commit**
```bash
git add src/main/resources/db/migration/V10__add_product_image_embeddings.sql \
        src/test/java/com/lezh1n/goodminton_shop_api/repositories/ProductImageEmbeddingsMigrationTest.java
git commit -m "$(cat <<'EOF'
feat(db): add product_image_embeddings vector(768) table + HNSW cosine index

New Flyway V10 for image search. Separate from kb_chunks (1024). FK to
products ON DELETE CASCADE, no resource_id FK (avoids write-time races).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Internal image-list endpoint `GET /api/internal/products/{id}/images`
**Files:**
- Create: `src/main/java/com/lezh1n/goodminton_shop_api/dtos/response/ProductImageResponse.java`
- Modify: `src/main/java/com/lezh1n/goodminton_shop_api/controllers/InternalProductController.java` (inject `ResourceService` after line 26; add endpoint after line 63 `getPricing`; add imports)
- Test (create): `src/test/java/com/lezh1n/goodminton_shop_api/security/InternalProductImagesEndpointTest.java`

**Interfaces:**
- Consumes: `ResourceService.listByOwner(ResourceOwner.PRODUCT_THUMBNAIL, id)` → `List<ResourceResponse>` (`id`, `url`, `sortOrder`).
- Produces: `List<ProductImageResponse>` where `ProductImageResponse(Integer resourceId, String url, Integer sortOrder)` — RAW (no `ApiResponse` wrapper). Guarded by `X-Internal-Key`.

- [ ] **Step 1: Write the failing test** Create `InternalProductImagesEndpointTest.java` — full-context MockMvc test (mirrors `RecommendationSecurityTest`) proving the endpoint returns the images with a valid `X-Internal-Key` and 401 without it.
```java
package com.lezh1n.goodminton_shop_api.security;

import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.web.servlet.MockMvc;

import com.lezh1n.goodminton_shop_api.dtos.response.ResourceResponse;
import com.lezh1n.goodminton_shop_api.enums.ResourceOwner;
import com.lezh1n.goodminton_shop_api.services.ResourceService;

@SpringBootTest
@AutoConfigureMockMvc
@TestPropertySource(properties = "app.internal-api-key=test-internal-key")
class InternalProductImagesEndpointTest {

    @Autowired
    MockMvc mockMvc;

    @MockBean
    ResourceService resourceService;

    @Test
    void images_withValidKey_returns200AndImages() throws Exception {
        when(resourceService.listByOwner(ResourceOwner.PRODUCT_THUMBNAIL, 5))
                .thenReturn(List.of(
                        ResourceResponse.builder().id(9).url("http://img/9").sortOrder(0).build(),
                        ResourceResponse.builder().id(10).url("http://img/10").sortOrder(1).build()));

        mockMvc.perform(get("/api/internal/products/5/images")
                        .header("X-Internal-Key", "test-internal-key"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].resourceId").value(9))
                .andExpect(jsonPath("$[0].url").value("http://img/9"))
                .andExpect(jsonPath("$[0].sortOrder").value(0))
                .andExpect(jsonPath("$[1].resourceId").value(10));
    }

    @Test
    void images_withoutKey_returns401() throws Exception {
        mockMvc.perform(get("/api/internal/products/5/images"))
                .andExpect(status().isUnauthorized());
    }
}
```
- [ ] **Step 2: Run test to verify it fails** Run: `./mvnw test -Dtest=InternalProductImagesEndpointTest`  Expected: FAIL — `images_withValidKey_returns200AndImages` gets 404 (no handler for `/images` yet). (`images_withoutKey_returns401` already passes via `InternalAuthFilter`.)
- [ ] **Step 3: Write minimal implementation**
  Create `ProductImageResponse.java`:
```java
package com.lezh1n.goodminton_shop_api.dtos.response;

// Internal image-list item for RAG's ImageIndexer. Field names are a cross-repo contract.
public record ProductImageResponse(Integer resourceId, String url, Integer sortOrder) {
}
```
  Edit `InternalProductController.java` — add imports (after line 17 `import ...ProductRepository;`):
```java
import com.lezh1n.goodminton_shop_api.dtos.response.ProductImageResponse;
import com.lezh1n.goodminton_shop_api.enums.ResourceOwner;
import com.lezh1n.goodminton_shop_api.services.ResourceService;
```
  Add the `ResourceService` dependency (after line 26 `private final ProductRepository productRepository;`):
```java
    private final ResourceService resourceService;
```
  Add the endpoint (after `getPricing`, before the class closing brace at line 64):
```java
    // Read-only image list for RAG's ImageIndexer. RAW list (no ApiResponse wrapper),
    // guarded by X-Internal-Key via InternalAuthFilter.
    @GetMapping("/{id}/images")
    public List<ProductImageResponse> getImages(@PathVariable Integer id) {
        return resourceService.listByOwner(ResourceOwner.PRODUCT_THUMBNAIL, id).stream()
                .map(r -> new ProductImageResponse(r.getId(), r.getUrl(), r.getSortOrder()))
                .toList();
    }
```
- [ ] **Step 4: Run test to verify it passes** Run: `./mvnw test -Dtest=InternalProductImagesEndpointTest`  Expected: PASS (both methods).
- [ ] **Step 5: Commit**
```bash
git add src/main/java/com/lezh1n/goodminton_shop_api/dtos/response/ProductImageResponse.java \
        src/main/java/com/lezh1n/goodminton_shop_api/controllers/InternalProductController.java \
        src/test/java/com/lezh1n/goodminton_shop_api/security/InternalProductImagesEndpointTest.java
git commit -m "$(cat <<'EOF'
feat(internal): add GET /api/internal/products/{id}/images for RAG indexing

Returns [{resourceId, url, sortOrder}] via resourceService.listByOwner.
X-Internal-Key guarded; raw list per internal-endpoint convention.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Public hydration endpoint `GET /api/products/list-items?ids=1,2,3`
**Files:**
- Modify: `src/main/java/com/lezh1n/goodminton_shop_api/repositories/ResourceRepository.java` (add batch lookup after line 15; add `Collection` import)
- Modify: `src/main/java/com/lezh1n/goodminton_shop_api/services/ProductService.java` (add method + imports)
- Modify: `src/main/java/com/lezh1n/goodminton_shop_api/services/impl/ProductServiceImpl.java` (add `ResourceRepository` field after line 61; add `listItemsByIds`; add imports)
- Modify: `src/main/java/com/lezh1n/goodminton_shop_api/controllers/ProductController.java` (add endpoint after line 57 `getAllProducts`; add imports)
- Modify: `src/main/java/com/lezh1n/goodminton_shop_api/configurations/SecurityConfig.java` (add `"api/products/list-items"` to `GET_PUBLIC_ENDPOINTS` after line 58)
- Test (create): `src/test/java/com/lezh1n/goodminton_shop_api/services/impl/ProductServiceListItemsTest.java` (order + is_visible)
- Test (create): `src/test/java/com/lezh1n/goodminton_shop_api/security/ListItemsEndpointPublicTest.java` (public, 200 without token)

**Interfaces:**
- Consumes: `ProductRepository.findVisibleByIdInWithVariants(Collection<Integer>)` (existing, line 89–91 — is_visible filter + eager variants); `ResourceRepository.findByOwnerTypeAndOwnerIdInOrderBySortOrderAsc(ResourceOwner, Collection<Integer>)` (new); `ProductMapper.toListItemResponse(Product, String)` (existing, line 96).
- Produces: `ProductService.listItemsByIds(List<Integer> ids) → List<ProductListItemResponse>` (requested order preserved, hidden/missing dropped); `GET /api/products/list-items?ids=` → `ApiResponse<List<ProductListItemResponse>>` (public).

- [ ] **Step 1: Write the failing test** Create BOTH test files.
  `ProductServiceListItemsTest.java` — Mockito unit test: repo returns visible subset in scrambled order; assert the service re-orders to the requested id order and drops the hidden id; assert batched thumbnails are applied.
```java
package com.lezh1n.goodminton_shop_api.services.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.lezh1n.goodminton_shop_api.dtos.response.ProductListItemResponse;
import com.lezh1n.goodminton_shop_api.entities.Product;
import com.lezh1n.goodminton_shop_api.entities.Resources;
import com.lezh1n.goodminton_shop_api.enums.ResourceOwner;
import com.lezh1n.goodminton_shop_api.mappers.ProductMapper;
import com.lezh1n.goodminton_shop_api.repositories.ProductRepository;
import com.lezh1n.goodminton_shop_api.repositories.ResourceRepository;

@ExtendWith(MockitoExtension.class)
class ProductServiceListItemsTest {

    @Mock ProductRepository productRepository;
    @Mock ResourceRepository resourceRepository;
    @Mock ProductMapper productMapper;

    @InjectMocks ProductServiceImpl service;

    private static Product product(int id) {
        return Product.builder().id(id).name("p" + id).slug("p" + id).build();
    }

    @Test
    void listItemsByIds_preservesRequestedOrder_dropsHidden_appliesBatchedThumbnail() {
        // Requested [1,2,3]; product 2 is hidden so the visible query omits it,
        // and returns the survivors in a DIFFERENT order (3 then 1).
        when(productRepository.findVisibleByIdInWithVariants(List.of(1, 2, 3)))
                .thenReturn(List.of(product(3), product(1)));
        when(resourceRepository.findByOwnerTypeAndOwnerIdInOrderBySortOrderAsc(
                eq(ResourceOwner.PRODUCT_THUMBNAIL), any()))
                .thenReturn(List.of(
                        Resources.builder().ownerId(1).url("http://t/1").sortOrder(0).build(),
                        Resources.builder().ownerId(1).url("http://t/1-b").sortOrder(1).build(),
                        Resources.builder().ownerId(3).url("http://t/3").sortOrder(0).build()));
        when(productMapper.toListItemResponse(any(Product.class), any()))
                .thenAnswer(inv -> ProductListItemResponse.builder()
                        .id(((Product) inv.getArgument(0)).getId())
                        .thumbnailUrl(inv.getArgument(1))
                        .build());

        List<ProductListItemResponse> result = service.listItemsByIds(List.of(1, 2, 3));

        // Order = requested order minus hidden(2): [1, 3]
        assertThat(result).extracting(ProductListItemResponse::getId).containsExactly(1, 3);
        // First row per owner (sort_order asc) is the thumbnail.
        assertThat(result.get(0).getThumbnailUrl()).isEqualTo("http://t/1");
        assertThat(result.get(1).getThumbnailUrl()).isEqualTo("http://t/3");
    }

    @Test
    void listItemsByIds_emptyInput_returnsEmpty() {
        assertThat(service.listItemsByIds(List.of())).isEmpty();
    }
}
```
  `ListItemsEndpointPublicTest.java` — full-context MockMvc test proving the endpoint is public (200 with no auth token).
```java
package com.lezh1n.goodminton_shop_api.security;

import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.servlet.MockMvc;

import com.lezh1n.goodminton_shop_api.services.ProductService;

@SpringBootTest
@AutoConfigureMockMvc
class ListItemsEndpointPublicTest {

    @Autowired
    MockMvc mockMvc;

    @MockBean
    ProductService productService;

    @Test
    void listItems_isPublic_returns200WithoutToken() throws Exception {
        when(productService.listItemsByIds(anyList())).thenReturn(List.of());

        mockMvc.perform(get("/api/products/list-items?ids=1,2,3"))
                .andExpect(status().isOk());
    }
}
```
- [ ] **Step 2: Run test to verify it fails** Run: `./mvnw test -Dtest=ProductServiceListItemsTest,ListItemsEndpointPublicTest`  Expected: FAIL — does not compile (`ProductServiceImpl.listItemsByIds` and `ResourceRepository.findByOwnerTypeAndOwnerIdInOrderBySortOrderAsc` do not exist yet).
- [ ] **Step 3: Write minimal implementation**
  Edit `ResourceRepository.java` — add `import java.util.Collection;` (after line 3 `import java.util.List;`) and the batch method (after line 15):
```java
    List<Resources> findByOwnerTypeAndOwnerIdInOrderBySortOrderAsc(
            ResourceOwner ownerType, Collection<Integer> ownerIds);
```
  Edit `ProductService.java` — add imports and the method to the interface:
```java
import java.util.List;

import com.lezh1n.goodminton_shop_api.dtos.response.ProductListItemResponse;
```
```java
    List<ProductListItemResponse> listItemsByIds(List<Integer> ids);
```
  Edit `ProductServiceImpl.java` — add imports (with the existing `java.util.*` imports and the `dtos.response` / `repositories` groups):
```java
import java.util.LinkedHashMap;

import com.lezh1n.goodminton_shop_api.dtos.response.ProductListItemResponse;
import com.lezh1n.goodminton_shop_api.repositories.ResourceRepository;
```
  Add the `ResourceRepository` field (after line 61 `private final ResourceService resourceService;`):
```java
    private final ResourceRepository resourceRepository;
```
  Add the method (e.g. after `getAllProducts`):
```java
    @Override
    public List<ProductListItemResponse> listItemsByIds(List<Integer> ids) {
        if (ids == null || ids.isEmpty()) {
            return List.of();
        }
        // H8: findVisibleByIdInWithVariants filters is_visible server-side + eager-loads variants.
        List<Product> visible = productRepository.findVisibleByIdInWithVariants(ids);
        if (visible.isEmpty()) {
            return List.of();
        }
        List<Integer> visibleIds = visible.stream().map(Product::getId).toList();
        // Batch thumbnail lookup (avoid N+1): one query; first row per owner (sort_order asc) = thumbnail.
        Map<Integer, String> thumbByProduct = new HashMap<>();
        resourceRepository
                .findByOwnerTypeAndOwnerIdInOrderBySortOrderAsc(ResourceOwner.PRODUCT_THUMBNAIL, visibleIds)
                .forEach(r -> thumbByProduct.putIfAbsent(r.getOwnerId(), r.getUrl()));
        Map<Integer, Product> byId = new LinkedHashMap<>();
        visible.forEach(p -> byId.put(p.getId(), p));
        // Preserve the requested id order; drop ids that were hidden/missing.
        return ids.stream()
                .map(byId::get)
                .filter(Objects::nonNull)
                .map(p -> productMapper.toListItemResponse(p, thumbByProduct.get(p.getId())))
                .toList();
    }
```
  Edit `ProductController.java` — add imports (with the existing imports):
```java
import java.util.List;

import com.lezh1n.goodminton_shop_api.dtos.response.ProductListItemResponse;
```
  Add the endpoint (after `getAllProducts`, before `updateProduct`):
```java
    @GetMapping("/list-items")
    public ApiResponse<List<ProductListItemResponse>> listItemsByIds(@RequestParam("ids") List<Integer> ids) {
        return ApiResponse.<List<ProductListItemResponse>>builder()
                .result(productService.listItemsByIds(ids))
                .build();
    }
```
  Edit `SecurityConfig.java` — add to `GET_PUBLIC_ENDPOINTS` (after line 58 `"api/products/{productId}/recommendations",`):
```java
            "api/products/list-items",
```
- [ ] **Step 4: Run test to verify it passes** Run: `./mvnw test -Dtest=ProductServiceListItemsTest,ListItemsEndpointPublicTest`  Expected: PASS (order preserved, hidden dropped, thumbnails batched, endpoint public).
- [ ] **Step 5: Commit**
```bash
git add src/main/java/com/lezh1n/goodminton_shop_api/repositories/ResourceRepository.java \
        src/main/java/com/lezh1n/goodminton_shop_api/services/ProductService.java \
        src/main/java/com/lezh1n/goodminton_shop_api/services/impl/ProductServiceImpl.java \
        src/main/java/com/lezh1n/goodminton_shop_api/controllers/ProductController.java \
        src/main/java/com/lezh1n/goodminton_shop_api/configurations/SecurityConfig.java \
        src/test/java/com/lezh1n/goodminton_shop_api/services/impl/ProductServiceListItemsTest.java \
        src/test/java/com/lezh1n/goodminton_shop_api/security/ListItemsEndpointPublicTest.java
git commit -m "$(cat <<'EOF'
feat(products): add public GET /api/products/list-items hydration endpoint

Order-preserving, is_visible-filtered (H8), batched thumbnail lookup.
Public for guest image search. Reuses toListItemResponse.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Publish image-changed event on upload + delete
**Files:**
- Modify: `src/main/java/com/lezh1n/goodminton_shop_api/services/impl/ProductServiceImpl.java` (`uploadProductImage` lines 149–155, `deleteProductImage` lines 157–160; add `Resources` import; reuse `ResourceRepository` field from Task 3)
- Test (create): `src/test/java/com/lezh1n/goodminton_shop_api/services/impl/ProductServiceImageEventTest.java`

**Interfaces:**
- Consumes: `ApplicationEventPublisher.publishEvent(...)` (existing field `events`); `ResourceRepository.findById(Integer)` (to resolve the owning product id on delete); `ProductChangedEvent.updated(Integer, Set<String>)` (existing).
- Produces: an `AFTER_COMMIT` RabbitMQ `product.updated` message with `fieldsChanged = {"images"}` (the RAG consumer's re-index marker) on both upload and delete.

- [ ] **Step 1: Write the failing test** Create `ProductServiceImageEventTest.java` — Mockito unit test capturing the published event.
```java
package com.lezh1n.goodminton_shop_api.services.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.web.multipart.MultipartFile;

import com.lezh1n.goodminton_shop_api.dtos.response.ResourceResponse;
import com.lezh1n.goodminton_shop_api.entities.Resources;
import com.lezh1n.goodminton_shop_api.enums.ResourceOwner;
import com.lezh1n.goodminton_shop_api.events.ProductChangedEvent;
import com.lezh1n.goodminton_shop_api.repositories.ProductRepository;
import com.lezh1n.goodminton_shop_api.repositories.ResourceRepository;
import com.lezh1n.goodminton_shop_api.services.ResourceService;

@ExtendWith(MockitoExtension.class)
class ProductServiceImageEventTest {

    @Mock ProductRepository productRepository;
    @Mock ResourceRepository resourceRepository;
    @Mock ResourceService resourceService;
    @Mock ApplicationEventPublisher events;
    @Mock MultipartFile file;

    @InjectMocks ProductServiceImpl service;

    @Test
    void uploadProductImage_publishesImagesUpdatedEvent() {
        when(productRepository.existsById(7)).thenReturn(true);
        when(resourceService.upload(ResourceOwner.PRODUCT_THUMBNAIL, 7, file))
                .thenReturn(ResourceResponse.builder().id(99).build());

        service.uploadProductImage(7, file);

        ArgumentCaptor<ProductChangedEvent> captor = ArgumentCaptor.forClass(ProductChangedEvent.class);
        verify(events).publishEvent(captor.capture());
        assertThat(captor.getValue().action()).isEqualTo("updated");
        assertThat(captor.getValue().productId()).isEqualTo(7);
        assertThat(captor.getValue().fieldsChanged()).containsExactly("images");
    }

    @Test
    void deleteProductImage_publishesImagesUpdatedEventForOwningProduct() {
        lenient().when(resourceRepository.findById(99)).thenReturn(Optional.of(
                Resources.builder().id(99).ownerId(42)
                        .ownerType(ResourceOwner.PRODUCT_THUMBNAIL).build()));

        service.deleteProductImage(99);

        verify(resourceService).delete(99);
        ArgumentCaptor<ProductChangedEvent> captor = ArgumentCaptor.forClass(ProductChangedEvent.class);
        verify(events).publishEvent(captor.capture());
        assertThat(captor.getValue().action()).isEqualTo("updated");
        assertThat(captor.getValue().productId()).isEqualTo(42);
        assertThat(captor.getValue().fieldsChanged()).containsExactly("images");
    }
}
```
- [ ] **Step 2: Run test to verify it fails** Run: `./mvnw test -Dtest=ProductServiceImageEventTest`  Expected: FAIL — `Wanted but not invoked: events.publishEvent(...)` (upload/delete currently publish no event; delete does not resolve the owning product id).
- [ ] **Step 3: Write minimal implementation** Edit `ProductServiceImpl.java`. Add the `Resources` import (with the entities import group):
```java
import com.lezh1n.goodminton_shop_api.entities.Resources;
```
  Replace `uploadProductImage` (lines 149–155):
```java
    @Override
    public ResourceResponse uploadProductImage(Integer productId, MultipartFile file) {
        if (!productRepository.existsById(productId)) {
            throw new AppException(ErrorCode.PRODUCT_NOT_FOUND);
        }
        ResourceResponse uploaded = resourceService.upload(ResourceOwner.PRODUCT_THUMBNAIL, productId, file);
        // Re-index marker for RAG image search (decision #4). "images" is the consumer contract.
        events.publishEvent(ProductChangedEvent.updated(productId, Set.of("images")));
        return uploaded;
    }
```
  Replace `deleteProductImage` (lines 157–160) — resolve the owning product id before deleting so the event carries it:
```java
    @Override
    public void deleteProductImage(Integer imageId) {
        Resources resource = resourceRepository.findById(imageId)
                .orElseThrow(() -> new AppException(ErrorCode.RESOURCE_NOT_FOUND));
        resourceService.delete(imageId);
        // Guard: only PRODUCT_THUMBNAIL resources map to a product. A variant/review
        // resource id would carry a non-product ownerId and re-index the wrong product.
        if (resource.getOwnerType() == ResourceOwner.PRODUCT_THUMBNAIL) {
            events.publishEvent(ProductChangedEvent.updated(resource.getOwnerId(), Set.of("images")));
        }
    }
```
- [ ] **Step 4: Run test to verify it passes** Run: `./mvnw test -Dtest=ProductServiceImageEventTest`  Expected: PASS (both upload and delete publish `updated` with `{"images"}`).
- [ ] **Step 5: Commit**
```bash
git add src/main/java/com/lezh1n/goodminton_shop_api/services/impl/ProductServiceImpl.java \
        src/test/java/com/lezh1n/goodminton_shop_api/services/impl/ProductServiceImageEventTest.java
git commit -m "$(cat <<'EOF'
feat(products): publish image-changed event on upload/delete for RAG re-index

Both paths now emit ProductChangedEvent.updated(productId, {"images"}) over
the product.* binding so RAG re-indexes image embeddings. Delete resolves
the owning product id from the resource before removal.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification
- [ ] Run the full changed surface: `./mvnw test -Dtest=ProductImageEmbeddingsMigrationTest,InternalProductImagesEndpointTest,ProductServiceListItemsTest,ListItemsEndpointPublicTest,ProductServiceImageEventTest`  Expected: PASS.
- [ ] Full suite (matches CI `mvn -B verify`): `./mvnw test`  Expected: PASS (requires local Postgres/Redis/RabbitMQ from `docker-compose.infra.yml`).
