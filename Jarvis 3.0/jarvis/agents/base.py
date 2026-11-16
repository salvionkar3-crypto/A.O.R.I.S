"""Base definitions and interfaces for specialized agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from jarvis.core.enums import AgentRole
from jarvis.memory.models import MemoryItem
from jarvis.providers.base import BaseLLMProvider, CompletionMessage


class AgentInput(BaseModel):
    """Input provided to an agent during execution."""
    user_request: str
    session_id: str
    memories: List[MemoryItem] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, str]] = Field(default_factory=list)


class AgentOutput(BaseModel):
    """Structured response from an agent."""
    agent_role: AgentRole
    response_text: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for specialized agents."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """The role identifier for this agent."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System instructions tailored to this agent's specialty."""
        pass

    def build_messages(self, agent_input: AgentInput) -> List[CompletionMessage]:
        """Construct prompt message sequence including relevant persistent memories."""
        messages: List[CompletionMessage] = [
            CompletionMessage(role="system", content=self.system_prompt)
        ]

        # Inject bounded relevant memories into system context
        if agent_input.memories:
            mem_text = "\n".join(
                f"- [{m.category.value}] {m.content} (Status: {m.fact_status.value}, Confidence: {m.confidence})"
                for m in agent_input.memories
            )
            messages.append(
                CompletionMessage(
                    role="system",
                    content=f"Relevant persistent memory records:\n{mem_text}",
                )
            )

        # Include recent conversation turns if present
        for h in agent_input.history[-5:]:
            messages.append(CompletionMessage(role=h.get("role", "user"), content=h.get("content", "")))

        # Finally, append user's immediate request
        messages.append(CompletionMessage(role="user", content=agent_input.user_request))
        return messages

    @abstractmethod
    async def run(self, agent_input: AgentInput) -> AgentOutput:
        """Execute agent reasoning and return structured output."""
        pass
