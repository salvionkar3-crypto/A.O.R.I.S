"""Automated tests for AI provider integration, error handling, retries, and health probes."""

import logging
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from jarvis.config import Settings
from jarvis.core.exceptions import (
    AuthenticationError,
    MalformedResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from jarvis.providers.base import (
    CompletionMessage,
    ProviderStatusCode,
)
from jarvis.providers.factory import create_llm_provider
from jarvis.providers.nemotron import NemotronProvider
from jarvis.providers.offline import OfflineProvider
from jarvis.utils.logging import setup_logger


@pytest.mark.asyncio
async def test_offline_provider_deterministic():
    """Verify offline provider produces coherent, deterministic responses without network."""
    provider = OfflineProvider()
    health = await provider.check_health()
    assert health.status == ProviderStatusCode.OFFLINE_FALLBACK_ACTIVE
    assert health.is_live is False
    assert health.offline_warning is not None

    # Conversation test
    msg = [CompletionMessage(role="user", content="Hello A.O.R.I.S., who are you?")]
    res = await provider.generate_response(msg)
    assert "A.O.R.I.S." in res.content
    assert res.is_live is False
    assert res.provider_name == "offline_fallback"

    # Planning test
    plan_msg = [CompletionMessage(role="user", content="Please plan this complex task.")]
    plan_res = await provider.generate_response(plan_msg)
    assert "execution plan" in plan_res.content.lower()


@pytest.mark.asyncio
async def test_provider_factory_prioritization():
    """Verify provider factory prioritizes Nebius Token Factory with eligible NVIDIA open model."""
    # 1. Nebius key provided
    nebius_settings = Settings(
        nebius_api_key="mock_nebius_secret_key_12345",
        nebius_default_model="nebius/nvidia/nemotron-3-super-120b-a12b",
    )
    provider1 = create_llm_provider(nebius_settings)
    assert isinstance(provider1, NemotronProvider)
    assert provider1.provider_name == "nebius_token_factory"
    assert provider1.default_model == "nebius/nvidia/nemotron-3-super-120b-a12b"
    assert provider1.is_live_provider is True

    # 2. NVIDIA key provided
    nvidia_settings = Settings(
        nvidia_api_key="nvapi-mock-nvidia-secret",
        nvidia_default_model="nvidia/nemotron-4-340b-instruct",
    )
    provider2 = create_llm_provider(nvidia_settings)
    assert isinstance(provider2, NemotronProvider)
    assert provider2.provider_name == "nvidia_nim"
    assert provider2.default_model == "nvidia/nemotron-4-340b-instruct"

    # 3. Neither provided -> offline fallback
    empty_settings = Settings(nebius_api_key=None, nvidia_api_key=None)
    provider3 = create_llm_provider(empty_settings)
    assert isinstance(provider3, OfflineProvider)
    assert provider3.provider_name == "offline_fallback"
    assert provider3.is_live_provider is False


@pytest.mark.asyncio
async def test_nemotron_missing_credentials():
    """Verify NemotronProvider raises AuthenticationError when API key is missing."""
    provider = NemotronProvider(api_key=None)
    with pytest.raises(AuthenticationError) as exc_info:
        await provider.generate_response([CompletionMessage(role="user", content="Hi")])
    assert "Missing API key" in str(exc_info.value)


@pytest.mark.asyncio
async def test_nemotron_successful_response():
    """Test response parsing on successful 200 OK from OpenAI-compatible endpoint."""
    mock_payload = {
        "id": "chatcmpl-nebius-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "nebius/nvidia/nemotron-3-super-120b-a12b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I am Nemotron on Nebius Token Factory.",
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 15, "total_tokens": 27}
    }

    provider = NemotronProvider(
        api_key="mock_secret_key",
        model="nebius/nvidia/nemotron-3-super-120b-a12b",
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = AsyncMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        res = await provider.generate_response([CompletionMessage(role="user", content="Hello")])
        assert res.content == "Hello! I am Nemotron on Nebius Token Factory."
        assert res.model == "nebius/nvidia/nemotron-3-super-120b-a12b"
        assert res.usage["total_tokens"] == 27
        assert res.is_live is True


@pytest.mark.asyncio
async def test_nemotron_auth_failure_no_repeated_retries():
    """Verify permanent 401 Unauthorized raises AuthenticationError immediately without retrying."""
    provider = NemotronProvider(api_key="invalid_key", max_retries=3)

    call_count = 0
    with patch("httpx.AsyncClient.post") as mock_post:
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = AsyncMock(spec=httpx.Response)
            mock_resp.status_code = 401
            mock_resp.text = "Unauthorized: Invalid API key"
            err = httpx.HTTPStatusError("401", request=AsyncMock(), response=mock_resp)
            mock_resp.raise_for_status.side_effect = err
            return mock_resp

        mock_post.side_effect = side_effect

        with pytest.raises(AuthenticationError):
            await provider.generate_response([CompletionMessage(role="user", content="Hello")])

        # Must have failed immediately on attempt 1 without wastefully retrying
        assert call_count == 1


@pytest.mark.asyncio
async def test_nemotron_model_not_found_no_retries():
    """Verify permanent 404 Model Not Found fails immediately without retrying."""
    provider = NemotronProvider(api_key="valid_key", model="non-existent-model", max_retries=3)

    call_count = 0
    with patch("httpx.AsyncClient.post") as mock_post:
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = AsyncMock(spec=httpx.Response)
            mock_resp.status_code = 404
            mock_resp.text = "Model not found"
            err = httpx.HTTPStatusError("404", request=AsyncMock(), response=mock_resp)
            mock_resp.raise_for_status.side_effect = err
            return mock_resp

        mock_post.side_effect = side_effect

        with pytest.raises(MalformedResponseError):
            await provider.generate_response([CompletionMessage(role="user", content="Hello")])

        assert call_count == 1


@pytest.mark.asyncio
async def test_nemotron_rate_limit_bounded_retries():
    """Verify HTTP 429 retries with backoff and raises RateLimitError when exhausted."""
    provider = NemotronProvider(api_key="test_key", max_retries=2)

    call_count = 0
    with patch("httpx.AsyncClient.post") as mock_post:
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = AsyncMock(spec=httpx.Response)
            mock_resp.status_code = 429
            mock_resp.text = "Rate limit exceeded"
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="429", request=AsyncMock(), response=mock_resp
            )
            return mock_resp

        mock_post.side_effect = side_effect

        with pytest.raises(RateLimitError):
            await provider.generate_response([CompletionMessage(role="user", content="Hello")])

        # Should have attempted max_retries times
        assert call_count == 2


@pytest.mark.asyncio
async def test_nemotron_timeout_error():
    """Verify timeout exception retries and raises ProviderTimeoutError."""
    provider = NemotronProvider(api_key="test_key", max_retries=2)

    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Connection timed out")):
        with pytest.raises(ProviderTimeoutError):
            await provider.generate_response([CompletionMessage(role="user", content="Hello")])


@pytest.mark.asyncio
async def test_nemotron_malformed_response():
    """Verify malformed JSON without choices raises MalformedResponseError."""
    provider = NemotronProvider(api_key="test_key")

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = AsyncMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"error": "unexpected format"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with pytest.raises(MalformedResponseError):
            await provider.generate_response([CompletionMessage(role="user", content="Hello")])


@pytest.mark.asyncio
async def test_live_inference_health_probe_success():
    """Verify check_health performs real probe completion and reports live status."""
    provider = NemotronProvider(
        api_key="valid_token_12345",
        model="nebius/nvidia/nemotron-3-super-120b-a12b",
    )

    probe_payload = {
        "id": "probe-1",
        "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = AsyncMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = probe_payload
        mock_post.return_value = mock_resp

        health = await provider.check_health()
        assert health.status == ProviderStatusCode.LIVE_INFERENCE_AVAILABLE
        assert health.is_live is True
        assert health.latency_ms is not None
        assert "verified successfully" in health.message.lower()


@pytest.mark.asyncio
async def test_live_inference_health_probe_auth_failure():
    """Verify check_health detects 401/403 and reports AUTHENTICATION_FAILED honestly."""
    provider = NemotronProvider(api_key="bad_token")

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = AsyncMock(spec=httpx.Response)
        mock_resp.status_code = 401
        mock_resp.text = "Invalid bearer token"
        mock_post.return_value = mock_resp

        health = await provider.check_health()
        assert health.status == ProviderStatusCode.AUTHENTICATION_FAILED
        assert health.is_live is False
        assert "Authentication failed" in health.message


@pytest.mark.asyncio
async def test_secret_redaction_in_logging():
    """Verify logger redacts API keys and tokens from log messages."""
    logger = setup_logger("test_secret_logger")

    class LogCaptureHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(self.format(record))

    cap = LogCaptureHandler()
    logger.addHandler(cap)

    secret_key = "nvapi-secret123456789abcdef"
    logger.info(f"Connecting with API key: {secret_key}")

    assert len(cap.records) > 0
    formatted_msg = cap.records[-1]
    assert secret_key not in formatted_msg
    assert "[REDACTED_API_KEY]" in formatted_msg or "[REDACTED]" in formatted_msg
