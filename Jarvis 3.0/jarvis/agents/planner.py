"""Planner agent for decomposing complex tasks into structured, dependency-aware steps."""

import json
import re
from typing import Any, Dict, List, Optional
from jarvis.agents.base import AgentInput, AgentOutput, BaseAgent
from jarvis.core.enums import AgentRole


class PlannerAgent(BaseAgent):
    """
    Decomposes multi-step goals into structured plans with dependencies,
    agent assignment, tool calls, and clear verification criteria.
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.PLANNER

    @property
    def system_prompt(self) -> str:
        return (
            "You are the JARVIS Task Planner Agent.\n"
            "Your role is to break down complex user requests into discrete, ordered execution steps.\n"
            "For each step, specify:\n"
            "- 'step_id': integer (1, 2, ...)\n"
            "- 'description': clear objective of the step\n"
            "- 'agent': one of 'research', 'coding', 'computer', 'conversation'\n"
            "- 'tool_name': optional name of tool to invoke (e.g. 'file_writer', 'file_reader', 'calculator')\n"
            "- 'requires_confirmation': true if modifying files or performing consequential actions\n"
            "- 'completion_criteria': explicit condition to verify success\n"
            "Return a valid JSON array of step objects, or a clear numbered list."
        )

    def _extract_tool_args_from_desc(self, tool_name: Optional[str], desc: str) -> Dict[str, Any]:
        """Infer default tool arguments from step description if not provided in JSON."""
        if not tool_name:
            return {}
        if tool_name == "calculator":
            match = re.search(r"([\d\s\+\-\*\/\(\)\.\%\^]+)", desc)
            expr = match.group(1).strip() if match else desc
            return {"expression": expr}
        elif tool_name == "file_writer":
            file_match = re.search(r"(?:to|file|named)\s+([^\s,;]+\.[a-zA-Z0-9]+)", desc)
            target = file_match.group(1) if file_match else "output.txt"
            content_match = re.search(r"(?:content|with|containing)\s+['\"]?(.+?)['\"]?$", desc)
            content = content_match.group(1) if content_match else "{{previous_step_result}}"
            return {"file_path": target, "content": content}
        elif tool_name == "file_reader":
            file_match = re.search(r"(?:file|from|path)\s+([^\s,;]+)", desc)
            target = file_match.group(1) if file_match else desc.split()[-1]
            return {"file_path": target}
        elif tool_name == "file_list":
            dir_match = re.search(r"(?:in|directory)\s+([^\s,;]+)", desc)
            target = dir_match.group(1) if dir_match else "."
            return {"directory_path": target}
        elif tool_name == "research_search":
            q_match = re.search(r"(?:for|about|search)\s+['\"]?(.+?)['\"]?$", desc)
            q = q_match.group(1) if q_match else desc
            return {"query": q}
        return {}

    def _parse_plan_steps(self, text: str, user_request: str = "") -> List[Dict[str, Any]]:
        """Parse structured steps from LLM response (JSON or numbered list fallback)."""
        # Attempt to parse json block
        json_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list) and len(parsed) > 0:
                    for s in parsed:
                        if "tool_name" in s and s["tool_name"] and "arguments" not in s:
                            s["arguments"] = self._extract_tool_args_from_desc(s["tool_name"], s.get("description", ""))
                    return parsed
            except Exception:
                pass

        # Fallback heuristic parser for compound requests or numbered steps
        steps: List[Dict[str, Any]] = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # Check if text contains numbered lines
        numbered_lines = [l for l in lines if re.match(r"^\d+[\.\)]\s+", l)]
        
        # If no numbered lines but request has "and then" or "then", split on "then"
        if not numbered_lines and user_request and any(w in user_request.lower() for w in [" and then ", " then ", ";"]):
            sub_clauses = re.split(r"\s+and\s+then\s+|\s+then\s+|;\s*", user_request, flags=re.IGNORECASE)
            numbered_lines = [f"{i+1}. {c.strip()}" for i, c in enumerate(sub_clauses) if c.strip()]

        step_idx = 1
        for line_str in (numbered_lines if numbered_lines else lines):
            clean_desc = re.sub(r"^\d+[\.\)]\s+", "", line_str).strip()
            if not clean_desc:
                continue

            tool = None
            requires_conf = False
            lower_desc = clean_desc.lower()
            if any(w in lower_desc for w in ["write", "create file", "save to"]):
                tool = "file_writer"
                requires_conf = True
            elif any(w in lower_desc for w in ["read", "inspect file", "show content"]):
                tool = "file_reader"
            elif any(w in lower_desc for w in ["calc", "compute", "what is"]):
                tool = "calculator"
            elif any(w in lower_desc for w in ["list files", "ls", "directory"]):
                tool = "file_list"
            elif any(w in lower_desc for w in ["research", "search docs", "find in docs"]):
                tool = "research_search"

            args = self._extract_tool_args_from_desc(tool, clean_desc) if tool else {}

            steps.append({
                "step_id": step_idx,
                "description": clean_desc,
                "agent": "computer" if tool else "research",
                "tool_name": tool,
                "arguments": args,
                "requires_confirmation": requires_conf,
                "completion_criteria": f"Successfully executed step: {clean_desc}",
            })
            step_idx += 1

        if not steps:
            steps.append({
                "step_id": 1,
                "description": user_request or "Execute requested task",
                "agent": "conversation",
                "tool_name": None,
                "arguments": {},
                "requires_confirmation": False,
                "completion_criteria": "Task fulfilled",
            })
        return steps

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        messages = self.build_messages(agent_input)
        response = await self.provider.generate_response(messages=messages, temperature=0.1)
        steps = self._parse_plan_steps(response.content, user_request=agent_input.user_request)

        return AgentOutput(
            agent_role=self.role,
            response_text=response.content,
            metadata={
                "steps": steps,
                "total_steps": len(steps),
                "model": response.model,
            },
        )

    async def replan(
        self,
        original_goal: str,
        failed_step: Dict[str, Any],
        failure_reason: str,
        completed_steps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Formulate a bounded recovery plan when an execution step fails.
        Provides both LLM-assisted replanning and deterministic fallback recovery.
        """
        replan_prompt = (
            f"You are replanning for task: '{original_goal}'.\n"
            f"Completed Steps: {len(completed_steps)}\n"
            f"Failed Step: {failed_step.get('description')} (Tool: {failed_step.get('tool_name')})\n"
            f"Failure Reason: {failure_reason}\n"
            "Provide 1 recovery step in JSON format or numbered list to overcome or safely address this failure."
        )

        recovery_steps: List[Dict[str, Any]] = []
        try:
            agent_in = AgentInput(user_request=replan_prompt)
            messages = self.build_messages(agent_in)
            resp = await self.provider.generate_response(messages=messages, temperature=0.1)
            recovery_steps = self._parse_plan_steps(resp.content, user_request=replan_prompt)
        except Exception:
            pass

        if not recovery_steps or (len(recovery_steps) == 1 and not recovery_steps[0].get("tool_name")):
            # Deterministic domain-specific recovery heuristics
            failed_tool = failed_step.get("tool_name")
            next_id = failed_step.get("step_id", 1) + 1
            if failed_tool == "file_reader":
                # If file not found, list directory to locate alternative
                recovery_steps = [{
                    "step_id": next_id,
                    "description": "List directory contents to locate correct filename",
                    "agent": "computer",
                    "tool_name": "file_list",
                    "arguments": {"directory_path": "."},
                    "requires_confirmation": False,
                    "completion_criteria": "Directory contents listed to diagnose missing file",
                }]
            elif failed_tool == "calculator":
                # Fallback arithmetic cleanup
                raw_expr = failed_step.get("arguments", {}).get("expression", "")
                cleaned_expr = re.sub(r"[^\d\+\-\*\/\(\)\.\s]", "", raw_expr)
                recovery_steps = [{
                    "step_id": next_id,
                    "description": f"Retry calculation with sanitized formula: {cleaned_expr}",
                    "agent": "computer",
                    "tool_name": "calculator",
                    "arguments": {"expression": cleaned_expr or "0"},
                    "requires_confirmation": False,
                    "completion_criteria": "Sanitized formula evaluated",
                }]
            else:
                recovery_steps = [{
                    "step_id": next_id,
                    "description": f"Explain execution limitation honestly: {failure_reason}",
                    "agent": "conversation",
                    "tool_name": None,
                    "arguments": {},
                    "requires_confirmation": False,
                    "completion_criteria": "Reported honest failure",
                }]

        return recovery_steps

