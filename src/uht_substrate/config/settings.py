"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable loading."""

    model_config = SettingsConfigDict(
        env_prefix="UHT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # UHT Factory API
    api_base_url: str = "https://factory.universalhex.org/api/v1"
    api_timeout: int = 30
    cache_ttl: int = 3600

    # Neo4j Database
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(default="uhtsubstrate123")
    neo4j_database: str = "neo4j"

    # Reasoning Configuration
    confidence_threshold: float = 0.7
    max_inference_depth: int = 3
    context_relevance_window_hours: int = 168  # 1 week

    # MCP Server
    server_name: str = "UHT Substrate Agent"
    server_host: str = "localhost"
    server_port: int = 8765

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
