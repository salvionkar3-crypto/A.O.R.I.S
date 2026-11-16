"""Provider abstraction module for JARVIS."""

from jarvis.providers.base import (
    BaseLLMProvider,
    CompletionMessage,
    CompletionResponse,
    MultiModelProviderInfo,
    MultiModelStatus,
    ProviderHealth,
    ProviderRole,
    ProviderStatusCode,
)
from jarvis.providers.factory import create_llm_provider
from jarvis.providers.nemotron import NemotronProvider
from jarvis.providers.offline import OfflineProvider
from jarvis.providers.registry import MultiModelRegistry
from jarvis.providers.specialists import (
    AnthropicCodingProvider,
    GeminiVisionProvider,
    OpenAIReasoningProvider,
)

__all__ = [
    "BaseLLMProvider",
    "CompletionMessage",
    "CompletionResponse",
    "MultiModelProviderInfo",
    "MultiModelStatus",
    "ProviderHealth",
    "ProviderRole",
    "ProviderStatusCode",
    "NemotronProvider",
    "OfflineProvider",
    "OpenAIReasoningProvider",
    "AnthropicCodingProvider",
    "GeminiVisionProvider",
    "MultiModelRegistry",
    "create_llm_provider",
]

