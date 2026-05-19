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
    retrieval_top_k: int = 5
    min_query_length: int = 2

    # LLM
    llm_temperature: float = 0.3
    llm_timeout_seconds: float = 120.0

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

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        if not (self.postgres_user and self.postgres_password):
            raise ValueError(
                "Need DATABASE_URL or POSTGRES_USER + POSTGRES_PASSWORD"
            )
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
            raise ValueError(
                "Need RABBITMQ_URL or RABBITMQ_USER + RABBITMQ_PASSWORD"
            )
        return (
            f"amqp://{quote_plus(self.rabbitmq_user)}:"
            f"{quote_plus(self.rabbitmq_password)}@"
            f"{self.rabbitmq_host}:{self.rabbitmq_port}/"
        )


settings = Settings()  # type: ignore[call-arg]
