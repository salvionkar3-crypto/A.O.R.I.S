"""Computer agent for safe tool execution under permission control."""

import json
from typing import Any, Dict, Optional
from jarvis.agents.base import AgentInput, AgentOutput, BaseAgent
from jarvis.core.enums import AgentRole
from jarvis.providers.base import BaseLLMProvider
from jarvis.tools.base import ToolResult
from jarvis.tools.registry import ToolRegistry


class ComputerAgent(BaseAgent):
    """
    Executes approved system, file, and computational operations via ToolRegistry.
    Strictly enforces permission boundaries and handles confirmation gates.
    """

    def __init__(self, provider: BaseLLMProvider, tool_registry: ToolRegistry) -> None:
        super().__init__(provider)
        self.tool_registry = tool_registry

    @property
    def role(self) -> AgentRole:
        return AgentRole.COMPUTER

    @property
    def system_prompt(self) -> str:
        tools_summary = json.dumps(self.tool_registry.list_tools(), indent=2)
        return (
            "You are the JARVIS Computer Execution Agent.\n"
            "Your role is to map task requests to approved tools and interpret results.\n"
            f"Available tools and schemas:\n{tools_summary}\n"
            "Never attempt unauthorized, non-registered, or out-of-boundary actions."
        )

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_confirmed: bool = False,
    ) -> ToolResult:
        """Execute a tool through the registry."""
        return await self.tool_registry.execute_tool(
            name=tool_name,
            arguments=arguments,
            user_confirmed=user_confirmed,
        )

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        # Check if context already specifies tool and arguments
        target_tool = agent_input.context.get("tool_name")
        target_args = agent_input.context.get("tool_args", {})
        user_confirmed = agent_input.context.get("user_confirmed", False)

        if target_tool:
            tool_res = await self.execute_tool(
                tool_name=target_tool,
                arguments=target_args,
                user_confirmed=user_confirmed,
            )
            response_msg = (
                f"Tool '{target_tool}' executed successfully: {tool_res.data}"
                if tool_res.success
                else f"Tool '{target_tool}' execution error: {tool_res.error}"
            )
            return AgentOutput(
                agent_role=self.role,
                response_text=response_msg,
                tool_calls=[{"tool": target_tool, "args": target_args, "result": tool_res.model_dump()}],
                metadata={"tool_result": tool_res.model_dump()},
            )

        messages = self.build_messages(agent_input)
        response = await self.provider.generate_response(messages=messages, temperature=0.0)
        return AgentOutput(
            agent_role=self.role,
            response_text=response.content,
            metadata={"model": response.model},
        )
