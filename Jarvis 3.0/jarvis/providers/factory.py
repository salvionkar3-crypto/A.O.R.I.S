"""Provider factory for JARVIS."""

from typing import Optional
from jarvis.config import Settings, get_settings
from jarvis.providers.base import BaseLLMProvider
from jarvis.providers.nemotron import NemotronProvider
from jarvis.providers.offline import OfflineProvider
from jarvis.utils.logging import setup_logger

logger = setup_logger(__name__)


def create_llm_provider(settings: Optional[Settings] = None) -> BaseLLMProvider:
    """
    Factory function to initialize the appropriate LLM provider.
    
    Priority:
    1. Nebius Token Factory (Primary Hackathon Route using eligible NVIDIA open model)
    2. Direct NVIDIA NIM Endpoint
    3. OfflineProvider (Deterministic fallback when credentials are absent)
    """
    cfg = settings or get_settings()

    if cfg.nebius_api_key and cfg.nebius_api_key.strip():
        logger.info(
            f"Initializing Nebius Token Factory provider with model '{cfg.nebius_default_model}' "
            f"at endpoint '{cfg.nebius_base_url}'"
        )
        return NemotronProvider(
            api_key=cfg.nebius_api_key.strip(),
            base_url=cfg.nebius_base_url,
            model=cfg.nebius_default_model,
            name="nebius_token_factory",
        )
    elif cfg.nvidia_api_key and cfg.nvidia_api_key.strip():
        logger.info(
            f"Initializing NVIDIA NIM provider with model '{cfg.nvidia_default_model}' "
            f"at endpoint '{cfg.nvidia_base_url}'"
        )
        return NemotronProvider(
            api_key=cfg.nvidia_api_key.strip(),
            base_url=cfg.nvidia_base_url,
            model=cfg.nvidia_default_model,
            name="nvidia_nim",
        )
    else:
        logger.info(
            "No remote API keys detected (NEBIUS_API_KEY / NVIDIA_API_KEY). "
            "Initializing OfflineProvider for local deterministic operation."
        )
        return OfflineProvider()
