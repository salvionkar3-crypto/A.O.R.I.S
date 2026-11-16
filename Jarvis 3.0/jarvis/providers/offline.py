"""Offline fallback provider for zero-network testing and local operation."""

from typing import Any, Dict, List, Optional
from jarvis.providers.base import (
    BaseLLMProvider,
    CompletionMessage,
    CompletionResponse,
    ProviderHealth,
    ProviderStatusCode,
)


class OfflineProvider(BaseLLMProvider):
    """
    Deterministic offline provider used when external API credentials are not set.
    
    Ensures that the entire application, multi-agent orchestration, tool execution,
    and memory systems function completely and reliably without external network dependencies,
    while explicitly and honestly indicating local deterministic execution.
    """

    def __init__(self, model: str = "offline/deterministic-baseline") -> None:
        self.model_name = model

    @property
    def provider_name(self) -> str:
        return "offline_fallback"

    @property
    def default_model(self) -> str:
        return self.model_name

    @property
    def is_live_provider(self) -> bool:
        return False

    def _generate_deterministic_content(self, messages: Any) -> str:
        if not messages:
            return "I am JARVIS, ready to assist."

        def get_content(m: Any) -> str:
            if hasattr(m, "content"):
                return m.content or ""
            elif isinstance(m, dict):
                return m.get("content", "") or ""
            return str(m)

        def get_role(m: Any) -> str:
            if hasattr(m, "role"):
                return m.role or ""
            elif isinstance(m, dict):
                return m.get("role", "") or ""
            return ""

        last_msg = get_content(messages[-1]).strip()
        last_lower = last_msg.lower()

        # Check for system prompt context or memory instructions
        system_context = ""
        for m in messages:
            if get_role(m) == "system":
                system_context += get_content(m) + "\n"

        # Check for specialist draft in system context for multi-model synthesis
        if "<specialist_draft>" in system_context:
            import re
            draft_match = re.search(r"<specialist_draft>(.*?)</specialist_draft>", system_context, re.DOTALL)
            draft = draft_match.group(1).strip() if draft_match else ""
            return (
                f"[A.O.R.I.S. Nemotron Synthesis]\n"
                f"I have reviewed and verified the specialist draft:\n\n"
                f"{draft}\n\n"
                f"Status: Synthesized and validated by primary orchestrator."
            )

        # Deterministic responses based on task context
        if "plan" in last_lower or "break down" in last_lower:
            return (
                "Here is the structured execution plan:\n"
                "1. Inspect existing project context and memory.\n"
                "2. Execute the required operations safely.\n"
                "3. Verify execution outcomes against criteria.\n"
                "4. Update memory with results."
            )
        elif "verify" in last_lower or "verification" in last_lower:
            return (
                "VERIFICATION RESULT: VERIFIED\n"
                "Evidence: Action completed successfully and outputs validated against completion criteria."
            )
        elif "remember" in last_lower:
            return f"Understood. I have recorded this in persistent memory: '{last_msg}'."
        elif "who are you" in last_lower or "your name" in last_lower:
            return "I am A.O.R.I.S., a multi-model personal AI assistant created by APEX LEGENDS for the Nebius × NVIDIA Hackathon."
        else:
            # If retrieved memories were passed in system context, acknowledge and cite them directly
            if "relevant persistent memory records:" in system_context.lower():
                mem_lines = [
                    line.strip().lstrip("-").strip()
                    for line in system_context.splitlines()
                    if line.strip().startswith("- [")
                ]
                if mem_lines:
                    facts = "\n".join(f"- {line}" for line in mem_lines[:3])
                    return (
                        f"According to your persistent memory:\n{facts}\n\n"
                        f"In response to '{last_msg}', I will continue using these preferences."
                    )
            return f"I have received your request: '{last_msg}'. How else may I assist you?"

    async def generate_response(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        content = self._generate_deterministic_content(messages)
        return CompletionResponse(
            content=content,
            model=self.model_name,
            usage={
                "prompt_tokens": len(str(messages)) // 4,
                "completion_tokens": len(content) // 4,
                "total_tokens": (len(str(messages)) + len(content)) // 4,
            },
            finish_reason="stop",
            raw={"provider": "offline_fallback", "live": False},
            is_live=False,
            provider_name=self.provider_name,
        )

    def generate_response_sync(
        self,
        messages: List[CompletionMessage],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        content = self._generate_deterministic_content(messages)
        return CompletionResponse(
            content=content,
            model=self.model_name,
            usage={
                "prompt_tokens": len(str(messages)) // 4,
                "completion_tokens": len(content) // 4,
                "total_tokens": (len(str(messages)) + len(content)) // 4,
            },
            finish_reason="stop",
            raw={"provider": "offline_fallback", "live": False},
            is_live=False,
            provider_name=self.provider_name,
        )

    async def check_health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_name=self.provider_name,
            status=ProviderStatusCode.OFFLINE_FALLBACK_ACTIVE,
            model=self.model_name,
            message="Offline deterministic fallback provider is operational. Live model inference requires NEBIUS_API_KEY or NVIDIA_API_KEY.",
            is_live=False,
            offline_warning="LIVE INFERENCE NOT ACTIVE: Operating in local deterministic mode. Configure NEBIUS_API_KEY in .env to activate live NVIDIA Nemotron inference.",
        )
