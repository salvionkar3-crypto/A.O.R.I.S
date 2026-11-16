"""NVIDIA Nemotron / Nebius Token Factory provider implementation."""

import asyncio
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
    ProviderHealth,
    ProviderStatusCode,
)
from jarvis.utils.logging import setup_logger

logger = setup_logger(__name__)


class NemotronProvider(BaseLLMProvider):
    """
    OpenAI-compatible adapter for Nebius Token Factory and NVIDIA NIM endpoints.
    
    Supports:
    - Primary route: Nebius Token Factory with eligible NVIDIA open models
      (e.g., nebius/nvidia/nemotron-3-super-120b-a12b, nvidia/nemotron-4-340b-instruct)
    - Direct route: NVIDIA NIM (integrate.api.nvidia.com/v1)
    - Bounded retries strictly for transient errors (429, 502, 503, 504, connection timeouts)
    - Immediate failure without retry on permanent errors (401, 403, 400, 404)
    - Secret sanitization and live inference verification probes
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.tokenfactory.nebius.com/v1",
        model: str = "nebius/nvidia/nemotron-3-super-120b-a12b",
        timeout: float = 45.0,
        max_retries: int = 3,
        name: str = "nebius_token_factory",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._provider_name = name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def default_model(self) -> str:
        return self.model_name

    @property
    def is_live_provider(self) -> bool:
        return True

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key or not self.api_key.strip():
            raise AuthenticationError(
                f"Missing API key for provider '{self.provider_name}'. "
                f"Please configure NEBIUS_API_KEY or NVIDIA_API_KEY in your environment or .env file."
            )
        return {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
            "User-Agent": "JARVIS-ApexLegends/1.0",
        }

    def _prepare_payload(
        self,
        messages: List[CompletionMessage],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model_name),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        return payload

    def _parse_response(self, data: Dict[str, Any]) -> CompletionResponse:
        try:
            choices = data.get("choices", [])
            if not choices:
                raise MalformedResponseError(f"Provider returned no choices: {data}")
            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            finish_reason = choice.get("finish_reason", "stop")
            usage = data.get("usage", {})
            model = data.get("model", self.model_name)

            return CompletionResponse(
                content=content,
                model=model,
                usage=usage,
                finish_reason=finish_reason,
                raw=data,
                is_live=True,
                provider_name=self.provider_name,
            )
        except Exception as e:
            if isinstance(e, MalformedResponseError):
                raise
            raise MalformedResponseError(f"Failed to parse provider response: {e}") from e

    def _classify_and_handle_status_error(
        self, exc: httpx.HTTPStatusError, attempt: int
    ) -> None:
        """
        Classify HTTP status codes and raise specific domain exceptions.
        Permanent errors (401, 403, 400, 404) are never retried.
        """
        status = exc.response.status_code
        err_text = exc.response.text[:300]

        # 1. Authentication failure — permanent, do not retry
        if status in (401, 403):
            raise AuthenticationError(
                f"Authentication failed (HTTP {status}): Check your {self.provider_name.upper()} API key."
            ) from exc

        # 2. Model not found or misconfigured endpoint — permanent, do not retry
        if status == 404:
            raise MalformedResponseError(
                f"Model or endpoint not found (HTTP 404): Model '{self.model_name}' on '{self.base_url}' does not exist."
            ) from exc

        # 3. Bad request / invalid schema — permanent, do not retry
        if status == 400:
            raise MalformedResponseError(
                f"Bad request parameters (HTTP 400): {err_text}"
            ) from exc

        # 4. Rate limiting (429) — transient, retry with backoff if attempts remain
        if status == 429:
            err = RateLimitError(f"Provider rate limit exceeded (HTTP 429): {err_text}")
            if attempt >= self.max_retries:
                raise err from exc
            return  # Allow loop to retry

        # 5. Gateway / server errors (500, 502, 503, 504) — transient
        if status >= 500:
            err = ProviderUnavailableError(
                f"Provider service unavailable (HTTP {status}): {err_text}"
            )
            if attempt >= self.max_retries:
                raise err from exc
            return  # Allow loop to retry

        # Unclassified error
        raise MalformedResponseError(f"Provider returned error status {status}: {err_text}") from exc

    async def generate_response(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        headers = self._get_headers()
        payload = self._prepare_payload(messages, temperature, max_tokens, **kwargs)
        endpoint = f"{self.base_url}/chat/completions"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return self._parse_response(data)

            except httpx.HTTPStatusError as e:
                # Classify error: raises immediately if permanent (401, 403, 404, 400)
                self._classify_and_handle_status_error(e, attempt)
                # If execution continues, it was a transient error (429, 5xx)
                last_error = e
                backoff = attempt * 1.5
                logger.warning(
                    f"Transient HTTP error ({e.response.status_code}) on attempt {attempt}/{self.max_retries}. "
                    f"Retrying in {backoff}s..."
                )
                await asyncio.sleep(backoff)

            except httpx.TimeoutException as e:
                last_error = ProviderTimeoutError(
                    f"Request to {endpoint} timed out after {self.timeout}s."
                )
                if attempt < self.max_retries:
                    backoff = attempt * 1.5
                    logger.warning(f"Timeout on attempt {attempt}/{self.max_retries}. Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                else:
                    raise last_error from e

            except httpx.RequestError as e:
                last_error = ProviderUnavailableError(f"Connection to provider failed: {e}")
                if attempt < self.max_retries:
                    backoff = attempt * 1.5
                    logger.warning(f"Network error on attempt {attempt}/{self.max_retries}. Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                else:
                    raise last_error from e

        if isinstance(last_error, httpx.HTTPStatusError):
            if last_error.response.status_code == 429:
                raise RateLimitError("Provider rate limit exceeded after maximum retries.")
            raise ProviderUnavailableError(f"Provider service unavailable after retries (HTTP {last_error.response.status_code}).")

        raise last_error or ProviderUnavailableError("Max retries exceeded without successful response.")

    def generate_response_sync(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        headers = self._get_headers()
        payload = self._prepare_payload(messages, temperature, max_tokens, **kwargs)
        endpoint = f"{self.base_url}/chat/completions"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return self._parse_response(data)

            except httpx.HTTPStatusError as e:
                self._classify_and_handle_status_error(e, attempt)
                last_error = e
                backoff = attempt * 1.5
                time.sleep(backoff)

            except httpx.TimeoutException as e:
                last_error = ProviderTimeoutError(
                    f"Sync request to {endpoint} timed out after {self.timeout}s."
                )
                if attempt < self.max_retries:
                    time.sleep(attempt * 1.5)
                else:
                    raise last_error from e

            except httpx.RequestError as e:
                last_error = ProviderUnavailableError(f"Connection to provider failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(attempt * 1.5)
                else:
                    raise last_error from e

        if isinstance(last_error, httpx.HTTPStatusError):
            if last_error.response.status_code == 429:
                raise RateLimitError("Provider rate limit exceeded after maximum retries.")
            raise ProviderUnavailableError(f"Provider service unavailable after retries (HTTP {last_error.response.status_code}).")

        raise last_error or ProviderUnavailableError("Max retries exceeded without successful response.")

    async def check_health(self) -> ProviderHealth:
        """
        Verify real live inference capability.
        Performs an actual minimal completion probe request against the live endpoint.
        """
        if not self.api_key or not self.api_key.strip():
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.LIVE_INFERENCE_NOT_CONFIGURED,
                model=self.model_name,
                message=f"No API key configured for {self.provider_name}. Set NEBIUS_API_KEY or NVIDIA_API_KEY to activate live inference.",
                is_live=False,
                endpoint_url=self.base_url,
                offline_warning="LIVE INFERENCE NOT ACTIVE: Configure API credentials to enable remote model reasoning.",
            )

        probe_messages = [CompletionMessage(role="user", content="Ping")]
        payload = self._prepare_payload(probe_messages, temperature=0.0, max_tokens=2)
        endpoint = f"{self.base_url}/chat/completions"
        start_time = time.time()

        try:
            headers = self._get_headers()
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                latency = round((time.time() - start_time) * 1000, 2)

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        return ProviderHealth(
                            provider_name=self.provider_name,
                            status=ProviderStatusCode.LIVE_INFERENCE_AVAILABLE,
                            model=self.model_name,
                            message=f"Live inference verified successfully ({latency}ms). Model is ready for autonomous orchestration.",
                            is_live=True,
                            endpoint_url=self.base_url,
                            latency_ms=latency,
                        )
                    else:
                        return ProviderHealth(
                            provider_name=self.provider_name,
                            status=ProviderStatusCode.MODEL_UNAVAILABLE_OR_MISCONFIGURED,
                            model=self.model_name,
                            message="Provider endpoint responded 200 OK but returned an empty response body.",
                            is_live=False,
                            endpoint_url=self.base_url,
                        )

                elif response.status_code in (401, 403):
                    return ProviderHealth(
                        provider_name=self.provider_name,
                        status=ProviderStatusCode.AUTHENTICATION_FAILED,
                        model=self.model_name,
                        message=f"Authentication failed (HTTP {response.status_code}). Verify that your API key is valid and not expired.",
                        is_live=False,
                        endpoint_url=self.base_url,
                    )

                elif response.status_code == 404:
                    return ProviderHealth(
                        provider_name=self.provider_name,
                        status=ProviderStatusCode.MODEL_UNAVAILABLE_OR_MISCONFIGURED,
                        model=self.model_name,
                        message=f"Model '{self.model_name}' was not found on '{self.base_url}' (HTTP 404).",
                        is_live=False,
                        endpoint_url=self.base_url,
                    )

                elif response.status_code == 429:
                    return ProviderHealth(
                        provider_name=self.provider_name,
                        status=ProviderStatusCode.PROVIDER_TEMPORARILY_UNAVAILABLE,
                        model=self.model_name,
                        message="Rate limit exceeded on provider endpoint (HTTP 429).",
                        is_live=False,
                        endpoint_url=self.base_url,
                    )

                else:
                    return ProviderHealth(
                        provider_name=self.provider_name,
                        status=ProviderStatusCode.PROVIDER_TEMPORARILY_UNAVAILABLE,
                        model=self.model_name,
                        message=f"Provider endpoint returned HTTP {response.status_code}: {response.text[:200]}",
                        is_live=False,
                        endpoint_url=self.base_url,
                    )

        except httpx.TimeoutException:
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.PROVIDER_TEMPORARILY_UNAVAILABLE,
                model=self.model_name,
                message=f"Connection timed out after 15s while contacting {self.base_url}.",
                is_live=False,
                endpoint_url=self.base_url,
            )

        except Exception as e:
            return ProviderHealth(
                provider_name=self.provider_name,
                status=ProviderStatusCode.PROVIDER_TEMPORARILY_UNAVAILABLE,
                model=self.model_name,
                message=f"Failed to reach provider endpoint: {e}",
                is_live=False,
                endpoint_url=self.base_url,
            )
