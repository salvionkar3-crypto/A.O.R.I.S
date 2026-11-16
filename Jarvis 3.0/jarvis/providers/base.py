"""Base provider protocol, health statuses, and schemas for model endpoints."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProviderStatusCode(str, Enum):
    """Standardized provider operational and health status codes."""
    LIVE_INFERENCE_AVAILABLE = "live_inference_available"
    LIVE_INFERENCE_NOT_CONFIGURED = "live_inference_not_configured"
    AUTHENTICATION_FAILED = "authentication_failed"
    PROVIDER_TEMPORARILY_UNAVAILABLE = "provider_temporarily_unavailable"
    MODEL_UNAVAILABLE_OR_MISCONFIGURED = "model_unavailable_or_misconfigured"
    OFFLINE_FALLBACK_ACTIVE = "offline_fallback_active"


class MultiModelStatus(str, Enum):
    """Accredited status for multi-model specialist adapters."""
    AVAILABLE_VERIFIED = "available_verified"
    CREDENTIALS_MISSING = "credentials_missing"
    AUTH_FAILED = "authentication_failed"
    RATE_LIMITED = "rate_limited"
    DISABLED = "disabled"


class ProviderRole(str, Enum):
    """Role specialization for multi-model architecture."""
    ORCHESTRATOR = "orchestrator"
    REASONING = "reasoning"
    CODING = "coding"
    VISION = "vision"


class MultiModelProviderInfo(BaseModel):
    """Status and configuration details for a specialist provider."""
    provider_id: str
    display_name: str
    role: ProviderRole
    status: MultiModelStatus
    model: str
    endpoint_url: Optional[str] = None
    is_live: bool = False
    description: str = ""
    error: Optional[str] = None
    last_verified: Optional[str] = None



class CompletionMessage(BaseModel):
    """Normalized chat message for model providers."""
    role: str
    content: str
    name: Optional[str] = None


class CompletionResponse(BaseModel):
    """Normalized completion response from model providers."""
    content: str
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = "stop"
    raw: Optional[Dict[str, Any]] = None
    is_live: bool = False
    provider_name: str = "unknown"


class ProviderHealth(BaseModel):
    """Health check status of a model provider."""
    provider_name: str
    status: ProviderStatusCode
    model: str
    message: str
    is_live: bool = False
    endpoint_url: Optional[str] = None
    offline_warning: Optional[str] = None
    latency_ms: Optional[float] = None


class BaseLLMProvider(ABC):
    """Abstract interface for LLM providers (Nebius Token Factory, NVIDIA NIM, offline fallback)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model identifier."""
        pass

    @property
    def is_live_provider(self) -> bool:
        """Indicate whether this provider calls a live remote API."""
        return False

    @abstractmethod
    async def generate_response(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Asynchronously generate response from provider endpoint."""
        pass

    @abstractmethod
    def generate_response_sync(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Synchronously generate response from provider endpoint."""
        pass

    @abstractmethod
    async def check_health(self) -> ProviderHealth:
        """
        Check endpoint connectivity, authentication, and verify actual inference capability.
        Must not report success merely because credentials exist or endpoints are reachable.
        """
        pass
