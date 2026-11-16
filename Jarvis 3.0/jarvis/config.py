import os
from pathlib import Path
from typing import Any, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General & Cloud Environment
    jarvis_env: str = Field(default="development", description="Environment mode (development/production)")
    jarvis_log_level: str = Field(default="INFO", description="Logging level")
    jarvis_host: str = Field(default="0.0.0.0", description="Host to bind server")
    jarvis_port: int = Field(default=8000, description="Port to bind server")

    # Security & Abuse Controls
    jarvis_api_key: Optional[str] = Field(
        default=None, description="Optional secret token for authenticating public API endpoints"
    )
    jarvis_cors_origins: List[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    jarvis_rate_limit_per_minute: int = Field(
        default=60, description="Maximum requests per minute per client IP"
    )
    jarvis_max_request_size_bytes: int = Field(
        default=10_485_760, description="Maximum allowed request payload size in bytes (default: 10MB)"
    )

    # Storage & Persistence
    jarvis_data_dir: Path = Field(
        default=Path("data"), description="Directory for persistent data"
    )
    jarvis_db_path: Path = Field(
        default=Path("data/jarvis.db"), description="Path to SQLite database"
    )

    # Memory Engine
    jarvis_memory_max_retrieval_results: int = Field(
        default=5, description="Max memories returned in single retrieval"
    )
    jarvis_memory_similarity_threshold: float = Field(
        default=0.3, description="Minimum relevance threshold for retrieval"
    )
    jarvis_enable_soft_delete: bool = Field(
        default=True, description="Enable soft-delete for audit trail"
    )

    # Storage & Persistence
    aoris_docs_dir: Path = Field(
        default=Path("data/generated_docs"), description="Directory for generated PDF and document artifacts"
    )

    # Nebius Token Factory (Primary Hackathon Route with eligible NVIDIA open model)
    nebius_api_key: Optional[str] = Field(default=None, description="Nebius API key")
    nebius_base_url: str = Field(
        default="https://api.tokenfactory.nebius.com/v1",
        description="Nebius Token Factory OpenAI-compatible API base URL",
    )
    nebius_default_model: str = Field(
        default="nebius/nvidia/nemotron-3-super-120b-a12b",
        description="Eligible NVIDIA open model on Nebius Token Factory",
    )

    # NVIDIA NIM (Direct NVIDIA endpoint)
    nvidia_api_key: Optional[str] = Field(default=None, description="NVIDIA NIM API key")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", description="NVIDIA API base URL"
    )
    nvidia_default_model: str = Field(
        default="nvidia/nemotron-4-340b-instruct", description="NVIDIA default model"
    )

    # Specialist Provider 1: OpenAI (Reasoning & Deep Analysis)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API base URL"
    )
    openai_model: str = Field(
        default="gpt-4o-mini", description="OpenAI model for reasoning specialist"
    )

    # Specialist Provider 2: Anthropic (Code & Technical Engineering)
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com/v1", description="Anthropic API base URL"
    )
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022", description="Anthropic model for coding specialist"
    )

    # Specialist Provider 3: Google (Multimodal Vision & Creative Assets)
    google_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    google_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Google Gemini API base URL",
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash", description="Google Gemini model for vision specialist"
    )

    @model_validator(mode="before")
    @classmethod
    def _read_cloud_platform_env(cls, data: Any) -> Any:
        """Honor standard PORT and HOST environment variables and API key aliases."""
        if isinstance(data, dict):
            if "jarvis_port" not in data and "PORT" in os.environ:
                try:
                    data["jarvis_port"] = int(os.environ["PORT"])
                except ValueError:
                    pass
            if "jarvis_host" not in data and "HOST" in os.environ:
                data["jarvis_host"] = os.environ["HOST"]
            # Support GEMINI_API_KEY alias for google_api_key
            if not data.get("google_api_key"):
                gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if gemini_key:
                    data["google_api_key"] = gemini_key
        return data

    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        """If jarvis_data_dir was customized and subpaths are still default, nest them inside data_dir."""
        default_db = Path("data/jarvis.db")
        if self.jarvis_db_path == default_db and self.jarvis_data_dir != Path("data"):
            self.jarvis_db_path = self.jarvis_data_dir / "jarvis.db"

        default_docs = Path("data/generated_docs")
        if self.aoris_docs_dir == default_docs and self.jarvis_data_dir != Path("data"):
            self.aoris_docs_dir = self.jarvis_data_dir / "generated_docs"
        return self

    def ensure_directories(self) -> None:
        """Ensure that necessary data, documents, and log directories exist."""
        self.jarvis_data_dir.mkdir(parents=True, exist_ok=True)
        self.aoris_docs_dir.mkdir(parents=True, exist_ok=True)
        if self.jarvis_db_path.parent:
            self.jarvis_db_path.parent.mkdir(parents=True, exist_ok=True)



# Singleton instance accessor
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create singleton application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the settings singleton (useful for tests)."""
    global _settings
    _settings = None
