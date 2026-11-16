"""Unit tests for configuration management."""

from pathlib import Path
from jarvis.config import Settings, get_settings, reset_settings


def test_default_settings():
    """Verify standard default configuration values."""
    reset_settings()
    settings = get_settings()
    assert settings.jarvis_env == "development"
    assert settings.jarvis_db_path == Path("data/jarvis.db")
    assert settings.jarvis_memory_max_retrieval_results == 5
    assert settings.jarvis_enable_soft_delete is True


def test_directory_creation(tmp_path: Path):
    """Verify that ensure_directories creates data directories."""
    data_dir = tmp_path / "custom_data"
    db_file = data_dir / "custom.db"
    settings = Settings(jarvis_data_dir=data_dir, jarvis_db_path=db_file)
    settings.ensure_directories()
    assert data_dir.exists()
    assert data_dir.is_dir()
