"""Specialist provider adapters for A.O.R.I.S. multi-model architecture.

Provides dedicated adapters for:
1. OpenAIReasoningProvider: Reasoning & Deep Analysis
2. AnthropicCodingProvider: Code & Technical Engineering
3. GeminiVisionProvider: Multimodal Vision & Creative Assets
"""

import asyncio
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional
import httpx

from jarvis.core.exceptions import (
    AuthenticationError,
    MalformedResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
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
from jarvis.utils.logging import setup_logger

logger = setup_logger(__name__)


class OpenAIReasoningProvider(BaseLLMProvider):
    """
    OpenAI adapter specializing in logical reasoning, formal analysis, and mathematics.
    Communicates via OpenAI /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key.strip() if api_key else None
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_status: MultiModelStatus = (
            MultiModelStatus.AVAILABLE_VERIFIED if self.api_key else MultiModelStatus.CREDENTIALS_MISSING
        )
        self._last_error: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "openai_reasoning"

    @property
    def default_model(self) -> str:
        return self.model_name

    @property
    def is_live_provider(self) -> bool:
        return bool(self.api_key)

    @property
    def role(self) -> ProviderRole:
        return ProviderRole.REASONING

    def get_status_info(self) -> MultiModelProviderInfo:
        status = MultiModelStatus.CREDENTIALS_MISSING if not self.api_key else self._last_status
        return MultiModelProviderInfo(
            provider_id="openai",
            display_name="OpenAI GPT",
            role=ProviderRole.REASONING,
            status=status,
            model=self.model_name,
            endpoint_url=self.base_url,
            is_live=self.is_live_provider and status == MultiModelStatus.AVAILABLE_VERIFIED,
            description="Reasoning & Deep Analysis Specialist",
            error=self._last_error,
        )

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise AuthenticationError("OPENAI_API_KEY is not configured.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AORIS-ApexLegends/1.0",
        }

    async def generate_response(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        headers = self._get_headers()
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": kwargs.get("model", self.model_name),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                self._last_error = str(exc)
                raise ProviderTimeoutError(f"OpenAI request timed out: {exc}") from exc
            except Exception as exc:
                self._last_error = str(exc)
                raise ProviderUnavailableError(f"OpenAI connection error: {exc}") from exc

            if resp.status_code in (401, 403):
                self._last_status = MultiModelStatus.AUTH_FAILED
                self._last_error = resp.text
                raise AuthenticationError(f"OpenAI authentication failed ({resp.status_code}): {resp.text}")
            elif resp.status_code == 429:
                self._last_status = MultiModelStatus.RATE_LIMITED
                self._last_error = resp.text
                raise RateLimitError(f"OpenAI rate limited: {resp.text}")
            elif resp.status_code != 200:
                self._last_error = resp.text
                raise ProviderUnavailableError(f"OpenAI returned error ({resp.status_code}): {resp.text}")

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            self._last_status = MultiModelStatus.AVAILABLE_VERIFIED
            return CompletionResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                is_live=True,
                provider_name=self.provider_name,
                raw=data,
            )

    def generate_response_sync(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        return asyncio.run(self.generate_response(messages, temperature, max_tokens, **kwargs))

    async def check_health(self) -> ProviderHealth:
        if not self.api_key:
            self._last_status = MultiModelStatus.CREDENTIALS_MISSING
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.LIVE_INFERENCE_NOT_CONFIGURED,
                model=self.model_name,
                message="OPENAI_API_KEY is not configured.",
                is_live=False,
                endpoint_url=self.base_url,
            )

        t0 = time.time()
        try:
            res = await self.generate_response(
                messages=[CompletionMessage(role="user", content="Ping. Respond with 'PONG'.")],
                max_tokens=8,
            )
            latency = (time.time() - t0) * 1000
            self._last_status = MultiModelStatus.AVAILABLE_VERIFIED
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.LIVE_INFERENCE_AVAILABLE,
                model=self.model_name,
                message=f"OpenAI live inference verified: '{res.content.strip()}'",
                is_live=True,
                endpoint_url=self.base_url,
                latency_ms=round(latency, 2),
            )
        except AuthenticationError as e:
            self._last_status = MultiModelStatus.AUTH_FAILED
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.AUTHENTICATION_FAILED,
                model=self.model_name,
                message=str(e),
                is_live=False,
                endpoint_url=self.base_url,
            )
        except RateLimitError as e:
            self._last_status = MultiModelStatus.RATE_LIMITED
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.PROVIDER_TEMPORARILY_UNAVAILABLE,
                model=self.model_name,
                message=str(e),
                is_live=False,
                endpoint_url=self.base_url,
            )
        except Exception as e:
            self._last_error = str(e)
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.PROVIDER_TEMPORARILY_UNAVAILABLE,
                model=self.model_name,
                message=str(e),
                is_live=False,
                endpoint_url=self.base_url,
            )


class AnthropicCodingProvider(BaseLLMProvider):
    """
    Anthropic Claude adapter specializing in code generation, refactoring, and technical engineering.
    Communicates via Anthropic /v1/messages endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1",
        model: str = "claude-3-5-sonnet-20241022",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key.strip() if api_key else None
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_status: MultiModelStatus = (
            MultiModelStatus.AVAILABLE_VERIFIED if self.api_key else MultiModelStatus.CREDENTIALS_MISSING
        )
        self._last_error: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "anthropic_coding"

    @property
    def default_model(self) -> str:
        return self.model_name

    @property
    def is_live_provider(self) -> bool:
        return bool(self.api_key)

    @property
    def role(self) -> ProviderRole:
        return ProviderRole.CODING

    def get_status_info(self) -> MultiModelProviderInfo:
        status = MultiModelStatus.CREDENTIALS_MISSING if not self.api_key else self._last_status
        return MultiModelProviderInfo(
            provider_id="anthropic",
            display_name="Anthropic Claude",
            role=ProviderRole.CODING,
            status=status,
            model=self.model_name,
            endpoint_url=self.base_url,
            is_live=self.is_live_provider and status == MultiModelStatus.AVAILABLE_VERIFIED,
            description="Code & Technical Engineering Specialist",
            error=self._last_error,
        )

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise AuthenticationError("ANTHROPIC_API_KEY is not configured.")
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "User-Agent": "AORIS-ApexLegends/1.0",
        }

    async def generate_response(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        headers = self._get_headers()
        url = f"{self.base_url}/messages"

        # Anthropic messages format: extract system message if present
        system_prompt: Optional[str] = None
        formatted_messages: List[Dict[str, str]] = []
        for m in messages:
            if m.role == "system" and system_prompt is None:
                system_prompt = m.content
            else:
                role = "assistant" if m.role == "assistant" else "user"
                formatted_messages.append({"role": role, "content": m.content})

        if not formatted_messages:
            formatted_messages = [{"role": "user", "content": "Hello"}]

        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model_name),
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                self._last_error = str(exc)
                raise ProviderTimeoutError(f"Anthropic request timed out: {exc}") from exc
            except Exception as exc:
                self._last_error = str(exc)
                raise ProviderUnavailableError(f"Anthropic connection error: {exc}") from exc

            if resp.status_code in (401, 403):
                self._last_status = MultiModelStatus.AUTH_FAILED
                self._last_error = resp.text
                raise AuthenticationError(f"Anthropic authentication failed ({resp.status_code}): {resp.text}")
            elif resp.status_code == 429:
                self._last_status = MultiModelStatus.RATE_LIMITED
                self._last_error = resp.text
                raise RateLimitError(f"Anthropic rate limited: {resp.text}")
            elif resp.status_code != 200:
                self._last_error = resp.text
                raise ProviderUnavailableError(f"Anthropic returned error ({resp.status_code}): {resp.text}")

            data = resp.json()
            content_blocks = data.get("content", [])
            text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
            content = "".join(text_parts)
            usage = data.get("usage", {})
            self._last_status = MultiModelStatus.AVAILABLE_VERIFIED
            return CompletionResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                is_live=True,
                provider_name=self.provider_name,
                raw=data,
            )

    def generate_response_sync(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        return asyncio.run(self.generate_response(messages, temperature, max_tokens, **kwargs))

    async def check_health(self) -> ProviderHealth:
        if not self.api_key:
            self._last_status = MultiModelStatus.CREDENTIALS_MISSING
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.LIVE_INFERENCE_NOT_CONFIGURED,
                model=self.model_name,
                message="ANTHROPIC_API_KEY is not configured.",
                is_live=False,
                endpoint_url=self.base_url,
            )

        t0 = time.time()
        try:
            res = await self.generate_response(
                messages=[CompletionMessage(role="user", content="Ping. Respond with 'PONG'.")],
                max_tokens=8,
            )
            latency = (time.time() - t0) * 1000
            self._last_status = MultiModelStatus.AVAILABLE_VERIFIED
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.LIVE_INFERENCE_AVAILABLE,
                model=self.model_name,
                message=f"Anthropic live inference verified: '{res.content.strip()}'",
                is_live=True,
                endpoint_url=self.base_url,
                latency_ms=round(latency, 2),
            )
        except AuthenticationError as e:
            self._last_status = MultiModelStatus.AUTH_FAILED
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.AUTHENTICATION_FAILED,
                model=self.model_name,
                message=str(e),
                is_live=False,
                endpoint_url=self.base_url,
            )
        except RateLimitError as e:
            self._last_status = MultiModelStatus.RATE_LIMITED
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.PROVIDER_TEMPORARILY_UNAVAILABLE,
                model=self.model_name,
                message=str(e),
                is_live=False,
                endpoint_url=self.base_url,
            )
        except Exception as e:
            self._last_error = str(e)
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.PROVIDER_TEMPORARILY_UNAVAILABLE,
                model=self.model_name,
                message=str(e),
                is_live=False,
                endpoint_url=self.base_url,
            )


class GeminiVisionProvider(BaseLLMProvider):
    """
    Google Gemini adapter specializing in multimodal vision, creative design, and asset synthesis.
    Communicates via Google Generative Language /v1beta/models/{model}:generateContent endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        model: str = "gemini-1.5-flash",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key.strip() if api_key else None
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_status: MultiModelStatus = (
            MultiModelStatus.AVAILABLE_VERIFIED if self.api_key else MultiModelStatus.CREDENTIALS_MISSING
        )
        self._last_error: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "gemini_vision"

    @property
    def default_model(self) -> str:
        return self.model_name

    @property
    def is_live_provider(self) -> bool:
        return bool(self.api_key)

    @property
    def role(self) -> ProviderRole:
        return ProviderRole.VISION

    def get_status_info(self) -> MultiModelProviderInfo:
        status = MultiModelStatus.CREDENTIALS_MISSING if not self.api_key else self._last_status
        return MultiModelProviderInfo(
            provider_id="google",
            display_name="Google Gemini",
            role=ProviderRole.VISION,
            status=status,
            model=self.model_name,
            endpoint_url=self.base_url,
            is_live=self.is_live_provider and status == MultiModelStatus.AVAILABLE_VERIFIED,
            description="Multimodal Vision & Creative Specialist",
            error=self._last_error,
        )

    async def generate_response(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        if not self.api_key:
            raise AuthenticationError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not configured.")

        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AORIS-ApexLegends/1.0",
        }

        # Convert messages to Gemini contents structure
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = "model" if m.role in ("assistant", "model") else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})

        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                self._last_error = str(exc)
                raise ProviderTimeoutError(f"Gemini request timed out: {exc}") from exc
            except Exception as exc:
                self._last_error = str(exc)
                raise ProviderUnavailableError(f"Gemini connection error: {exc}") from exc

            if resp.status_code in (400, 401, 403):
                # Google returns 400 for API_KEY_INVALID
                err_text = resp.text
                if "API_KEY_INVALID" in err_text or resp.status_code in (401, 403):
                    self._last_status = MultiModelStatus.AUTH_FAILED
                    self._last_error = err_text
                    raise AuthenticationError(f"Google Gemini authentication failed: {err_text}")
                else:
                    self._last_error = err_text
                    raise MalformedResponseError(f"Google Gemini bad request: {err_text}")
            elif resp.status_code == 429:
                self._last_status = MultiModelStatus.RATE_LIMITED
                self._last_error = resp.text
                raise RateLimitError(f"Google Gemini rate limited: {resp.text}")
            elif resp.status_code != 200:
                self._last_error = resp.text
                raise ProviderUnavailableError(f"Google Gemini returned error ({resp.status_code}): {resp.text}")

            data = resp.json()
            candidates = data.get("candidates", [])
            content = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                content = "".join(p.get("text", "") for p in parts if "text" in p)

            usage = data.get("usageMetadata", {})
            self._last_status = MultiModelStatus.AVAILABLE_VERIFIED
            return CompletionResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                is_live=True,
                provider_name=self.provider_name,
                raw=data,
            )

    def generate_response_sync(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        return asyncio.run(self.generate_response(messages, temperature, max_tokens, **kwargs))

    async def check_health(self) -> ProviderHealth:
        if not self.api_key:
            self._last_status = MultiModelStatus.CREDENTIALS_MISSING
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.LIVE_INFERENCE_NOT_CONFIGURED,
                model=self.model_name,
                message="GOOGLE_API_KEY is not configured.",
                is_live=False,
                endpoint_url=self.base_url,
            )

        t0 = time.time()
        try:
            res = await self.generate_response(
                messages=[CompletionMessage(role="user", content="Ping. Respond with 'PONG'.")],
                max_tokens=8,
            )
            latency = (time.time() - t0) * 1000
            self._last_status = MultiModelStatus.AVAILABLE_VERIFIED
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.LIVE_INFERENCE_AVAILABLE,
                model=self.model_name,
                message=f"Google Gemini live inference verified: '{res.content.strip()}'",
                is_live=True,
                endpoint_url=self.base_url,
                latency_ms=round(latency, 2),
            )
        except AuthenticationError as e:
            self._last_status = MultiModelStatus.AUTH_FAILED
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.AUTHENTICATION_FAILED,
                model=self.model_name,
                message=str(e),
                is_live=False,
                endpoint_url=self.base_url,
            )
        except RateLimitError as e:
            self._last_status = MultiModelStatus.RATE_LIMITED
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.PROVIDER_TEMPORARILY_UNAVAILABLE,
                model=self.model_name,
                message=str(e),
                is_live=False,
                endpoint_url=self.base_url,
            )
        except Exception as e:
            self._last_error = str(e)
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.PROVIDER_TEMPORARILY_UNAVAILABLE,
                model=self.model_name,
                message=str(e),
                is_live=False,
                endpoint_url=self.base_url,
            )
