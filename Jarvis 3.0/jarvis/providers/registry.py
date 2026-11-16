"""Multi-Model Provider Registry managing Orchestrator and Specialist Adapters."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from jarvis.config import Settings, get_settings
from jarvis.providers.base import (
    BaseLLMProvider,
    MultiModelProviderInfo,
    MultiModelStatus,
    ProviderHealth,
    ProviderRole,
    ProviderStatusCode,
)
from jarvis.providers.factory import create_llm_provider
from jarvis.providers.specialists import (
    AnthropicCodingProvider,
    GeminiVisionProvider,
    OpenAIReasoningProvider,
)
from jarvis.utils.logging import setup_logger

logger = setup_logger(__name__)


class MultiModelRegistry:
    """
    Registry for managing all multi-model providers in A.O.R.I.S.:
    - Primary Orchestrator & Final Synthesis: NVIDIA Nemotron (via Nebius Token Factory or NIM or Offline)
    - Reasoning Specialist: OpenAI (GPT)
    - Coding Specialist: Anthropic (Claude)
    - Vision Specialist: Google (Gemini)
    """

    def __init__(self, settings: Optional[Settings] = None, orchestrator: Optional[BaseLLMProvider] = None) -> None:
        self.settings = settings or get_settings()
        self.orchestrator = orchestrator or create_llm_provider(self.settings)

        # Initialize specialist adapters
        self.openai_provider = OpenAIReasoningProvider(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            model=self.settings.openai_model,
        )
        self.anthropic_provider = AnthropicCodingProvider(
            api_key=self.settings.anthropic_api_key,
            base_url=self.settings.anthropic_base_url,
            model=self.settings.anthropic_model,
        )
        self.gemini_provider = GeminiVisionProvider(
            api_key=self.settings.google_api_key,
            base_url=self.settings.google_base_url,
            model=self.settings.gemini_model,
        )

        self.specialists: Dict[str, BaseLLMProvider] = {
            "openai": self.openai_provider,
            "anthropic": self.anthropic_provider,
            "google": self.gemini_provider,
        }

    def get_orchestrator(self) -> BaseLLMProvider:
        """Return the primary orchestrator and synthesis provider (NVIDIA Nemotron / Offline)."""
        return self.orchestrator

    def get_specialist(self, provider_id: str) -> Optional[BaseLLMProvider]:
        """Return a specialist provider by ID ('openai', 'anthropic', 'google')."""
        return self.specialists.get(provider_id.lower().strip())

    def is_specialist_available(self, provider_id: str) -> bool:
        """Check if a specialist has valid credentials and is live."""
        sp = self.get_specialist(provider_id)
        if not sp or not sp.is_live_provider:
            return False
        if hasattr(sp, "get_status_info"):
            info = sp.get_status_info()
            return info.status in (MultiModelStatus.AVAILABLE_VERIFIED, MultiModelStatus.RATE_LIMITED)
        return True

    def resolve_provider(self, specialist_name: Optional[str] = None) -> BaseLLMProvider:
        """
        Resolve provider for a task.
        If a specialist is requested/routed and available, return it.
        Otherwise, gracefully fall back to the primary orchestrator (Nemotron or offline).
        """
        if specialist_name:
            norm = specialist_name.lower().strip()
            # Map aliases
            if norm in ("claude", "coding"):
                norm = "anthropic"
            elif norm in ("gpt", "reasoning", "math"):
                norm = "openai"
            elif norm in ("gemini", "vision", "creative"):
                norm = "google"

            sp = self.get_specialist(norm)
            if sp and sp.is_live_provider:
                logger.info(f"Dispatched sub-task to specialist provider '{sp.provider_name}' ({sp.default_model})")
                return sp
            else:
                logger.info(
                    f"Specialist '{specialist_name}' not available or unconfigured; "
                    f"gracefully falling back to Orchestrator '{self.orchestrator.provider_name}'."
                )

        return self.orchestrator

    def get_orchestrator_status_info(self) -> MultiModelProviderInfo:
        """Return standardized status info for primary orchestrator."""
        name = self.orchestrator.provider_name
        model = self.orchestrator.default_model
        is_live = self.orchestrator.is_live_provider

        if is_live:
            status = MultiModelStatus.AVAILABLE_VERIFIED
            display = f"NVIDIA Nemotron ({name})"
            desc = "Primary Orchestrator & Final Synthesis Model"
            endpoint = getattr(self.orchestrator, "base_url", None)
        else:
            status = MultiModelStatus.DISABLED if "offline" not in name else MultiModelStatus.CREDENTIALS_MISSING
            display = "NVIDIA Nemotron (Offline Fallback Active)"
            desc = "Primary Orchestrator (Deterministic offline baseline active)"
            endpoint = None

        return MultiModelProviderInfo(
            provider_id="nemotron",
            display_name=display,
            role=ProviderRole.ORCHESTRATOR,
            status=status,
            model=model,
            endpoint_url=endpoint,
            is_live=is_live,
            description=desc,
        )

    def get_all_statuses(self) -> Dict[str, MultiModelProviderInfo]:
        """Return dictionary of statuses for all 4 providers."""
        return {
            "nemotron": self.get_orchestrator_status_info(),
            "openai": self.openai_provider.get_status_info(),
            "anthropic": self.anthropic_provider.get_status_info(),
            "google": self.gemini_provider.get_status_info(),
        }

    async def check_all_health(self) -> Dict[str, ProviderHealth]:
        """Perform concurrent health check probes across all configured providers."""
        orch_health = await self.orchestrator.check_health()
        openai_health = await self.openai_provider.check_health()
        anthropic_health = await self.anthropic_provider.check_health()
        gemini_health = await self.gemini_provider.check_health()

        return {
            "nemotron": orch_health,
            "openai": openai_health,
            "anthropic": anthropic_health,
            "google": gemini_health,
        }
