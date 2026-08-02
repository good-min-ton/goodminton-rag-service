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
    rerank_mode: str = "bge"  # "bge" (cross-encoder service, mặc định) | "llm" (Qwen listwise)
    rerank_url: str | None = None  # bge-reranker service, dùng khi mode == "bge"
    rerank_top_n: int = 4

    # Chatbot conversation state (Redis) + display cards.
    # redis_url e.g. redis://:pass@redis:6379/0 ; None => stateless degrade.
    redis_url: str | None = None
    chat_state_ttl_seconds: int = 3600
    chat_display_products_max: int = 4

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

    # CORS — Phase 3 mở "*" cho test, Phase 6 sẽ restrict theo FRONTEND_URL
    cors_origins: list[str] = ["*"]

    # RabbitMQ — Phase 4 consumer
    rabbitmq_url: str | None = None
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str | None = None
    rabbitmq_password: str | None = None

    products_exchange: str = "goodminton.products"
    rag_product_queue: str = "rag.product.sync"
    rag_product_dlq: str = "rag.product.sync.dlq"

    # Chunking cho product text
    product_chunk_size: int = 500
    product_chunk_overlap: int = 50

    # Shop API internal endpoint
    shop_api_url: str = "http://shop-api:8080"
    internal_api_key: str | None = None
    # Central store whose inventory row prepare_order reads (env: CENTRAL_STORE_NAME)
    central_store_name: str = "Goodminton HQ - Di An"  # Store.name where is_central=true (verified 2026-07-26)

    # Image search — embed-service (SigLIP, port 8001) + pgvector
    embed_service_url: str = "http://localhost:8001"
    image_search_top_k: int = 12
    image_search_over_fetch_factor: int = 3  # over-fetch = top_k * this (H8)
    # Max cosine distance for an image match; products beyond this are dropped so
    # an unrelated query image returns nothing. 0/negative disables the filter.
    # Calibrated from live data (matches ~0.09, unrelated ~0.31+). Tunable via env.
    image_search_max_distance: float = 0.30
    image_max_upload_bytes: int = 8 * 1024 * 1024  # 8 MB outer cap

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
