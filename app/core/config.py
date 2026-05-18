from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str

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


settings = Settings()  # type: ignore[call-arg]
