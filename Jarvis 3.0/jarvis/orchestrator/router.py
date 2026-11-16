"""Central AgentRouter for intent classification and workflow selection."""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from jarvis.core.enums import AgentRole
from jarvis.providers.base import BaseLLMProvider


class RoutingDecision(BaseModel):
    """The outcome of task routing analysis."""
    intent: str
    primary_agent: AgentRole
    complexity: str = "low"  # low, medium, high
    required_capabilities: List[str] = Field(default_factory=list)
    workflow_type: str = "conversational"  # direct_tool, planned_steps, agent_execution, conversational
    requires_planning: bool = False
    suggested_tool: Optional[str] = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    rationale: str = ""
    specialist_model: Optional[str] = None  # e.g., 'openai', 'anthropic', 'google'
    requires_synthesis: bool = False


class AgentRouter:
    """
    Intelligently routes requests to specialized agents, multi-model specialists,
    or direct tools. NVIDIA Nemotron acts as primary orchestrator.
    """

    def __init__(self, provider: Optional[BaseLLMProvider] = None) -> None:
        self.provider = provider

    def route(self, user_request: str, specialist_override: Optional[str] = None) -> RoutingDecision:
        """Analyze user request and determine optimal routing and specialist strategy."""
        text = user_request.strip()
        lower = text.lower()

        # Handle explicit specialist override from user/CLI if provided
        forced_specialist = specialist_override.lower().strip() if specialist_override else None

        # 0. PDF / Document Generation Requests
        pdf_match = re.search(
            r"(?:create|generate|make|build|save|export)\s+(?:a\s+)?pdf(?:\s+(?:named|titled|called)\s+([^\s,;]+))?(?:\s+(?:with|containing|about|titled)\s+(.+))?",
            text,
            re.IGNORECASE,
        )
        if "pdf" in lower and (pdf_match or any(w in lower for w in ["pdf report", "generate pdf", "make pdf", "create pdf", "save as pdf"])):
            title = "A.O.R.I.S. Generated Report"
            content = text
            filename = None
            if pdf_match:
                extracted_name = pdf_match.group(1)
                extracted_content = pdf_match.group(2)
                if extracted_name and extracted_name.endswith(".pdf"):
                    filename = extracted_name
                    title = extracted_name.replace(".pdf", "").replace("_", " ").title()
                elif extracted_name:
                    title = extracted_name.replace("_", " ").title()
                if extracted_content:
                    content = extracted_content

            return RoutingDecision(
                intent="pdf_generation",
                primary_agent=AgentRole.COMPUTER,
                complexity="medium",
                required_capabilities=["pdf_generation", "document_delivery"],
                workflow_type="direct_tool",
                requires_planning=False,
                suggested_tool="pdf_generator",
                tool_args={"title": title, "content": content, "filename": filename},
                confidence=0.98,
                rationale="ReportLab PDF generation requested; routed directly to pdf_generator tool.",
            )

        # 1. Complex Multi-Step Planning (Check first to avoid partial single-tool capture)
        has_multi_step_keywords = any(w in lower for w in [
            "plan", "workflow", "multi-step", "first", "then", "step by step",
            "and then", "after that", "subtasks", "decompose"
        ])
        is_coding_request = any(w in lower for w in [
            "def ", "class ", "write code", "python script", "unit test",
            "debug", "refactor", "fix error", "implement function", "write a function",
            "write a script", "write a program", "code to"
        ])
        has_compound_actions = (
            not is_coding_request
            and (
                ("write to" in lower or "save to" in lower or "write a file" in lower or "save a file" in lower or "save file" in lower)
                and ("calculate" in lower or "research" in lower or "read" in lower)
            )
        )

        if (has_multi_step_keywords or has_compound_actions) and not (is_coding_request and not has_multi_step_keywords):
            return RoutingDecision(
                intent="multi_step_planning",
                primary_agent=AgentRole.PLANNER,
                complexity="high",
                required_capabilities=["multi_step_planning", "sequential_execution"],
                workflow_type="planned_steps",
                requires_planning=True,
                confidence=0.95,
                specialist_model=forced_specialist,
                requires_synthesis=bool(forced_specialist),
                rationale="Request requires structured decomposition, dependency tracking, and multi-agent coordination.",
            )


        # 2. Calculation Requests
        calc_match = re.search(
            r"(?:calculate|compute|what is|solve)\s+([\d\s\+\-\*\/\(\)\.\%\^]+)$",
            text,
            re.IGNORECASE,
        )
        if calc_match and any(c in calc_match.group(1) for c in "+-*/%^"):
            expr = calc_match.group(1).strip()
            return RoutingDecision(
                intent="calculation",
                primary_agent=AgentRole.COMPUTER,
                complexity="low",
                required_capabilities=["math"],
                workflow_type="direct_tool",
                requires_planning=False,
                suggested_tool="calculator",
                tool_args={"expression": expr},
                confidence=1.0,
                rationale="Direct deterministic arithmetic evaluation via safe AST.",
            )

        # 3. File Writing / Creation
        file_write_match = re.search(
            r"(?:write|create|save|append\s+to)\s+(?:a\s+)?file\s+(?:named\s+|called\s+|at\s+)?([^\s,;]+)(?:\s+(?:with|containing)\s+(?:content\s+)?(.+))?",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if file_write_match:
            path = file_write_match.group(1).strip().strip("'\"")
            content = file_write_match.group(2) or ""
            content = content.strip().strip("'\"")
            return RoutingDecision(
                intent="file_write",
                primary_agent=AgentRole.COMPUTER,
                complexity="medium",
                required_capabilities=["filesystem_write", "consequential_confirmation"],
                workflow_type="direct_tool",
                requires_planning=False,
                suggested_tool="file_writer",
                tool_args={"file_path": path, "content": content},
                confidence=0.95,
                rationale="Consequential filesystem write operation requiring confirmation gate.",
            )

        # 4. File Reading / Inspection
        file_read_match = re.search(
            r"(?:read|inspect|show|display|view)\s+(?:the\s+)?file\s+([^\s,;]+)",
            text,
            re.IGNORECASE,
        )
        if file_read_match:
            path = file_read_match.group(1).strip().strip("'\"")
            return RoutingDecision(
                intent="file_read",
                primary_agent=AgentRole.COMPUTER,
                complexity="low",
                required_capabilities=["filesystem_read"],
                workflow_type="direct_tool",
                requires_planning=False,
                suggested_tool="file_reader",
                tool_args={"file_path": path},
                confidence=0.95,
                rationale="Direct sandboxed file read within allowed roots.",
            )

        # 5. Directory Listing
        if any(p in lower for p in ["list files", "show directory", "ls in", "list directory"]):
            dir_path = "."
            dir_match = re.search(r"in\s+([^\s,;]+)", text, re.IGNORECASE)
            if dir_match:
                dir_path = dir_match.group(1).strip().strip("'\"")
            return RoutingDecision(
                intent="file_list",
                primary_agent=AgentRole.COMPUTER,
                complexity="low",
                required_capabilities=["filesystem_read"],
                workflow_type="direct_tool",
                requires_planning=False,
                suggested_tool="file_list",
                tool_args={"directory_path": dir_path},
                confidence=0.95,
                rationale="Directory inventory inspection.",
            )

        # 6. Explicit Memory Search
        if any(lower.startswith(p) for p in ["search memory", "find in memory", "lookup memory"]):
            query_match = re.sub(r"^(?:search\s+memory(?:\s+for)?|find\s+in\s+memory|lookup\s+memory)\s*", "", text, flags=re.IGNORECASE).strip()
            return RoutingDecision(
                intent="memory_search",
                primary_agent=AgentRole.COMPUTER,
                complexity="low",
                required_capabilities=["memory_retrieval", "fts5_search"],
                workflow_type="direct_tool",
                requires_planning=False,
                suggested_tool="memory_search",
                tool_args={"query": query_match or text},
                confidence=0.95,
                rationale="Explicit FTS5 persistent memory search.",
            )

        # 7. Explicit Memory Directives
        if any(lower.startswith(p) for p in ["remember", "please remember", "my preference is", "my favorite", "save memory"]):
            return RoutingDecision(
                intent="memory_store",
                primary_agent=AgentRole.CONVERSATION,
                complexity="low",
                required_capabilities=["memory_write", "provenance_tracking"],
                workflow_type="conversational",
                requires_planning=False,
                confidence=1.0,
                rationale="Explicit persistent knowledge extraction and storage.",
            )

        # 8. Research / Information Gathering
        if any(w in lower for w in ["research", "investigate", "find sources", "citations", "look up", "who is", "what is the latest"]):
            return RoutingDecision(
                intent="research",
                primary_agent=AgentRole.RESEARCH,
                complexity="medium",
                required_capabilities=["web_research", "document_synthesis", "source_citation"],
                workflow_type="agent_execution",
                requires_planning=False,
                specialist_model=forced_specialist,
                requires_synthesis=bool(forced_specialist),
                confidence=0.9,
                rationale="Evidence gathering, citation verification, and factual synthesis.",
            )

        # 9. Coding Inquiries (Specialist: Anthropic Claude)
        if is_coding_request or any(w in lower for w in ["def ", "class ", "write code", "python script", "unit test", "debug", "refactor", "fix error", "implement function"]):
            return RoutingDecision(
                intent="coding",
                primary_agent=AgentRole.CODING,
                complexity="medium",
                required_capabilities=["coding", "code_verification"],
                workflow_type="agent_execution",
                requires_planning=False,
                specialist_model=forced_specialist or "anthropic",
                requires_synthesis=True,
                confidence=0.92,
                rationale="Technical software engineering routed to Anthropic Claude specialist with Nemotron synthesis.",
            )

        # 10. Deep Reasoning & Logic Analysis (Specialist: OpenAI GPT)
        if any(w in lower for w in ["prove ", "proof", "formal logic", "deep reasoning", "reason step by step", "complex analysis", "solve riddle", "deduce"]):
            return RoutingDecision(
                intent="reasoning",
                primary_agent=AgentRole.CONVERSATION,
                complexity="high",
                required_capabilities=["deep_reasoning", "logical_deduction"],
                workflow_type="agent_execution",
                requires_planning=False,
                specialist_model=forced_specialist or "openai",
                requires_synthesis=True,
                confidence=0.90,
                rationale="Complex logical analysis routed to OpenAI Reasoning specialist with Nemotron synthesis.",
            )

        # 11. Multimodal Vision & Creative Assets (Specialist: Google Gemini)
        if any(w in lower for w in ["visualize", "ui mockup", "creative design", "color palette", "image prompt", "multimodal", "describe image"]):
            return RoutingDecision(
                intent="vision_creative",
                primary_agent=AgentRole.CONVERSATION,
                complexity="medium",
                required_capabilities=["multimodal_vision", "creative_generation"],
                workflow_type="agent_execution",
                requires_planning=False,
                specialist_model=forced_specialist or "google",
                requires_synthesis=True,
                confidence=0.88,
                rationale="Creative visual layout routed to Google Gemini specialist with Nemotron synthesis.",
            )

        # 12. Default: Conversational Assistance (NVIDIA Nemotron Direct)
        return RoutingDecision(
            intent="conversation",
            primary_agent=AgentRole.CONVERSATION,
            complexity="low",
            required_capabilities=["dialogue", "memory_retrieval"],
            workflow_type="conversational",
            requires_planning=False,
            specialist_model=forced_specialist,
            requires_synthesis=bool(forced_specialist),
            confidence=0.85,
            rationale="Context-aware natural dialogue orchestrated directly by NVIDIA Nemotron.",
        )


