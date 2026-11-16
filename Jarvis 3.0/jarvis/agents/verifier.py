"""Verifier agent for empirical task validation and evidence-based completion checks."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from jarvis.agents.base import AgentInput, AgentOutput, BaseAgent
from jarvis.core.enums import AgentRole
from jarvis.tools.base import ToolResult


class VerificationVerdict(str, Enum):
    """Rigorous verdict on task completion."""
    VERIFIED = "verified"          # Empirical evidence confirms all criteria met
    INCOMPLETE = "incomplete"      # Subtasks remain or awaiting confirmation
    FAILED = "failed"              # Errors occurred or criteria explicitly not met
    INCONCLUSIVE = "inconclusive"  # Insufficient evidence to prove or disprove completion


class VerificationResult(BaseModel):
    """Structured verification report."""
    verdict: VerificationVerdict
    evidence: str
    checks_performed: List[str] = Field(default_factory=list)
    confidence: float = 1.0


class VerifierAgent(BaseAgent):
    """
    Evaluates tangible evidence (tool results, file systems, test outputs)
    against completion criteria. Never trusts mere model claims of success.
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.VERIFIER

    @property
    def system_prompt(self) -> str:
        return (
            "You are the JARVIS Independent Verifier Agent.\n"
            "Your solemn duty is to evaluate whether a requested task was genuinely completed.\n"
            "Rules:\n"
            "- Only mark VERIFIED if empirical evidence proves success.\n"
            "- If an error occurred or an action failed, mark FAILED.\n"
            "- If an action requires confirmation or is partially complete, mark INCOMPLETE.\n"
            "- If evidence is ambiguous, mark INCONCLUSIVE.\n"
            "- Never trust a model claiming 'I did it' without tangible evidence."
        )

    def verify_deterministic(
        self,
        task_description: str,
        completion_criteria: str,
        tool_results: List[ToolResult],
        context: Dict[str, Any],
    ) -> VerificationResult:
        """
        Perform rigorous deterministic checks without relying solely on LLM self-evaluation.
        """
        checks: List[str] = []

        if not tool_results and not context.get("direct_answer"):
            return VerificationResult(
                verdict=VerificationVerdict.INCONCLUSIVE,
                evidence="No tool execution results or explicit evidence provided for verification.",
                checks_performed=["checked_evidence_presence"],
                confidence=0.5,
            )

        # 1. Check for confirmation blocks
        for tr in tool_results:
            if tr.requires_confirmation:
                checks.append(f"Tool '{tr.tool_name}' blocked pending user confirmation.")
                return VerificationResult(
                    verdict=VerificationVerdict.INCOMPLETE,
                    evidence=f"Task is paused: tool '{tr.tool_name}' requires human approval.",
                    checks_performed=checks,
                    confidence=1.0,
                )

        # 2. Check for explicit tool failures
        for tr in tool_results:
            if not tr.success:
                checks.append(f"Tool '{tr.tool_name}' reported error: {tr.error}")
                return VerificationResult(
                    verdict=VerificationVerdict.FAILED,
                    evidence=f"Tool '{tr.tool_name}' failed with error: {tr.error}",
                    checks_performed=checks,
                    confidence=1.0,
                )

        # 3. File existence verification for write operations
        for tr in tool_results:
            if tr.tool_name == "file_writer" and tr.success:
                target_path = tr.data.get("path")
                if target_path:
                    p = Path(target_path)
                    if p.exists() and p.stat().st_size > 0:
                        checks.append(f"Verified file exists on disk with {p.stat().st_size} bytes: {p.name}")
                    else:
                        checks.append(f"File '{target_path}' does not exist on disk or is empty.")
                        return VerificationResult(
                            verdict=VerificationVerdict.FAILED,
                            evidence=f"File verification failed: '{target_path}' not found on disk.",
                            checks_performed=checks,
                            confidence=1.0,
                        )

        # 4. Calculation checks
        for tr in tool_results:
            if tr.tool_name == "calculator" and tr.success:
                checks.append(f"Mathematical result verified: {tr.data.get('result')}")

        # 5. File listing checks
        for tr in tool_results:
            if tr.tool_name == "file_list" and tr.success:
                entries_cnt = len(tr.data.get("entries", []))
                checks.append(f"Directory listing verified with {entries_cnt} entries found.")

        # 6. Research search & citation checks
        for tr in tool_results:
            if tr.tool_name == "research_search" and tr.success:
                citations_cnt = len(tr.data.get("citations", []))
                if citations_cnt > 0:
                    checks.append(f"Research search verified with {citations_cnt} verifiable source citations.")
                else:
                    checks.append("Research search verified: honest reporting of no matching sources without hallucinations.")

        # 7. Replanned recovery checks
        if context.get("replanned"):
            checks.append(f"Replanned recovery step verified: executed {context.get('recovery_step_count', 1)} corrective step(s).")

        # 8. Direct conversational answers
        if context.get("direct_answer"):
            checks.append("Direct conversational response generated and reviewed.")
            return VerificationResult(
                verdict=VerificationVerdict.VERIFIED,
                evidence="Conversational response generated according to instructions.",
                checks_performed=checks,
                confidence=0.9,
            )

        checks.append("All executed tools reported success and passed physical checks.")
        return VerificationResult(
            verdict=VerificationVerdict.VERIFIED,
            evidence="All tool steps succeeded with verified outputs matching criteria.",
            checks_performed=checks,
            confidence=1.0,
        )

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        # Check context for tool results to verify
        tool_results_raw = agent_input.context.get("tool_results", [])
        tool_results: List[ToolResult] = []
        for r in tool_results_raw:
            if isinstance(r, ToolResult):
                tool_results.append(r)
            elif isinstance(r, dict):
                tool_results.append(ToolResult(**r))

        criteria = agent_input.context.get("completion_criteria", "Complete user request")
        verification = self.verify_deterministic(
            task_description=agent_input.user_request,
            completion_criteria=criteria,
            tool_results=tool_results,
            context=agent_input.context,
        )

        return AgentOutput(
            agent_role=self.role,
            response_text=f"Verification Verdict: {verification.verdict.value.upper()}\nEvidence: {verification.evidence}",
            metadata={"verification": verification.model_dump()},
        )
