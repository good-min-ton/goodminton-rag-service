from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database — option 1: DSN trực tiếp (dev local)
    # Option 2: từng field — compose pass riêng để tránh issue URL-encoding password
    database_url: str | None = None

    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str = "goodminton"

    # Ollama
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    llm_model: str = "qwen2.5:14b-instruct-q4_K_M"

    # Retrieval
    retrieval_top_k: int = 5  # chunks đưa vào ngữ cảnh LLM
    min_query_length: int = 2
    # Số candidate lấy về trước khi rerank (recall cao hơn top_k).
    retrieval_candidates: int = 12
    # Ngưỡng khoảng cách cosine để một product chunk đủ điều kiện làm thẻ gợi ý;
    # vượt ngưỡng bị coi là quá yếu, không hiển thị. <=0 => tắt lọc.
    card_max_distance: float = 0.62

    # Rerank các candidate sản phẩm để thẻ gợi ý sát câu hỏi hơn.
    rerank_enabled: bool = True
    rerank_mode: str = (
        "bge"  # "bge" (cross-encoder service, mặc định) | "llm" (Qwen listwise)
    )
    rerank_url: str | None = None  # bge-reranker service, dùng khi mode == "bge"
    rerank_top_n: int = 4

    # Chatbot conversation state (Redis) + display cards.
    # redis_url e.g. redis://:pass@redis:6379/0 ; None => stateless degrade.
    redis_url: str | None = None
    chat_state_ttl_seconds: int = 3600
    chat_display_products_max: int = 4

    # Chat streaming (SSE) — flag-gated; OFF = /chat unchanged
    chat_stream_enabled: bool = False
    chat_rate_window_seconds: int = 60
    chat_rate_max: int = 30

    # Similar Products
    similar_products_top_k: int = 5
    similar_products_max_limit: int = 50

    # LLM
    llm_temperature: float = 0.3
    llm_timeout_seconds: float = 120.0

    # LLM — auto product description (Feature B)
    description_model: str | None = None
    description_temperature: float = 0.5
    description_num_predict: int = 1536

    # CORS. Comma-separated, e.g.
    #   CORS_ORIGINS=https://goodminton.vercel.app,https://*.vercel.app
    # A plain string rather than list[str]: pydantic-settings parses a list field
    # from the environment as JSON, which makes for an awkward compose value.
    # Default "*" keeps every origin allowed, which is what the public tunnel
    # needs until the frontend has a settled domain.
    cors_origins: str = "*"

    # RabbitMQ — Phase 4 consumer
    rabbitmq_url: str | None = None
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str | None = None
    rabbitmq_password: str | None = None

    # Tên exchange cố ý không có ở đây: chỉ phía publish mới cần biết, mà phía
    # publish là shop-api. Consumer chỉ cần tên hàng đợi đã được bind sẵn.
    rag_product_queue: str = "rag.product.sync"
    rag_product_dlq: str = "rag.product.sync.dlq"

    # Chunking cho product text
    product_chunk_size: int = 500
    product_chunk_overlap: int = 50

    # Shop API internal endpoint
    shop_api_url: str = "http://shop-api:8080"
    internal_api_key: str | None = None
    # No central-store setting on purpose: shop-api flags the central store on
    # every inventory row it returns, which is the same source of truth its own
    # checkout uses. A name kept in config here had to be updated by hand
    # whenever a store was renamed or promoted, and drifting silently meant
    # every variant read as out of stock.

    # Image search — embed-service (SigLIP, port 8001) + pgvector
    embed_service_url: str = "http://localhost:8001"
    image_search_top_k: int = 12
    image_search_over_fetch_factor: int = 3  # over-fetch = top_k * this (H8)
    # Max cosine distance for an image match; products beyond this are dropped so
    # an unrelated query image returns nothing. 0/negative disables the filter.
    # Calibrated from live data (matches ~0.09, unrelated ~0.31+). Tunable via env.
    image_search_max_distance: float = 0.30
    image_max_upload_bytes: int = 8 * 1024 * 1024  # 8 MB outer cap

    # Text->image cross-modal distance follows a different distribution than
    # image->image (which image_search_max_distance=0.30 was calibrated on).
    # 0 = filter disabled until calibrated on live text->image data (F1 eval).
    text_search_max_distance: float = 0.0

    @property
    def cors_middleware_kwargs(self) -> dict:
        """Origin arguments for Starlette's CORSMiddleware.

        Entries may contain a `*` wildcard so Vercel preview deployments, whose
        subdomain changes per branch, can be allowed without listing each one.
        Starlette matches exact origins from a list and wildcards from a regex,
        so wildcard entries are compiled into one. It uses `fullmatch`, so the
        pattern cannot match a longer attacker-controlled origin.

        `*` in the list means any origin; it wins over anything else present,
        because a narrower entry alongside it would only look like a restriction.
        """
        import re

        entries = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        if not entries or "*" in entries:
            return {"allow_origins": ["*"]}

        exact = [o for o in entries if "*" not in o]
        wildcards = [o for o in entries if "*" in o]
        if not wildcards:
            return {"allow_origins": exact}
        # `[^.]*` keeps a wildcard to a single label: https://*.vercel.app must
        # not also match https://evil.attacker.vercel.app. Note shop-api reads
        # the same style of value through Spring's origin patterns, which expand
        # `*` to `.*` and so span any number of labels. This side is the stricter
        # of the two; list exact origins when the difference matters.
        pattern = "|".join(re.escape(o).replace(r"\*", r"[^.]*") for o in wildcards)
        return {"allow_origins": exact, "allow_origin_regex": pattern}

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        if not (self.postgres_user and self.postgres_password):
            raise ValueError("Need DATABASE_URL or POSTGRES_USER + POSTGRES_PASSWORD")
        return (
            f"postgresql://{quote_plus(self.postgres_user)}:"
            f"{quote_plus(self.postgres_password)}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def resolved_rabbitmq_url(self) -> str:
        if self.rabbitmq_url:
            return self.rabbitmq_url
        if not (self.rabbitmq_user and self.rabbitmq_password):
            raise ValueError("Need RABBITMQ_URL or RABBITMQ_USER + RABBITMQ_PASSWORD")
        return (
            f"amqp://{quote_plus(self.rabbitmq_user)}:"
            f"{quote_plus(self.rabbitmq_password)}@"
            f"{self.rabbitmq_host}:{self.rabbitmq_port}/"
        )


settings = Settings()  # type: ignore[call-arg]
