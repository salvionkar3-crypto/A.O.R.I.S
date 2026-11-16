"""Coding agent for code generation, inspection, debugging, and testing."""

from jarvis.agents.base import AgentInput, AgentOutput, BaseAgent
from jarvis.core.enums import AgentRole


class CodingAgent(BaseAgent):
    """
    Assists with software engineering, code inspection, and test creation.
    Adheres to robust engineering standards and never automatically executes arbitrary code.
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.CODING

    @property
    def system_prompt(self) -> str:
        return (
            "You are the JARVIS Senior Coding Agent.\n"
            "Responsibilities:\n"
            "- Assist with high quality, secure, maintainable Python code.\n"
            "- Include comprehensive type hints, docstrings, and error handling.\n"
            "- Treat all external or generated code as untrusted until verified.\n"
            "- Propose automated unit and integration tests for critical functionality."
        )

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        messages = self.build_messages(agent_input)
        response = await self.provider.generate_response(messages=messages, temperature=0.1)
        return AgentOutput(
            agent_role=self.role,
            response_text=response.content,
            metadata={"model": response.model},
        )
