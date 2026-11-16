"""Conversation agent for natural dialogue, explanations, and general assistance."""

from jarvis.agents.base import AgentInput, AgentOutput, BaseAgent
from jarvis.core.enums import AgentRole


class ConversationAgent(BaseAgent):
    """
    Handles natural conversation, clarification, and general dialogue.
    Seamlessly incorporates relevant persistent memories into responses
    without exposing unrelated personal data.
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.CONVERSATION

    @property
    def system_prompt(self) -> str:
        return (
            "You are JARVIS, an advanced, highly intelligent personal AI assistant developed "
            "by team APEX LEGENDS for the Nebius × NVIDIA Global AI Hackathon.\n"
            "Style guidelines:\n"
            "- Be concise, articulate, helpful, and natural.\n"
            "- If relevant persistent memories are provided, use them respectfully to personalize "
            "your answers without unnecessarily reciting internal IDs or metadata.\n"
            "- Never make unsupported claims about what you did; speak honestly about your actual actions."
        )

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        messages = self.build_messages(agent_input)
        response = await self.provider.generate_response(messages=messages, temperature=0.3)
        return AgentOutput(
            agent_role=self.role,
            response_text=response.content,
            metadata={"model": response.model, "usage": response.usage},
        )
