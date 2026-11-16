"""Central OrchestratorEngine coordinating memory, router, agents, tools, and verifier."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from jarvis.agents.base import AgentInput
from jarvis.agents.coding import CodingAgent
from jarvis.agents.computer import ComputerAgent
from jarvis.agents.conversation import ConversationAgent
from jarvis.agents.planner import PlannerAgent
from jarvis.agents.research import ResearchAgent
from jarvis.agents.verifier import (
    VerificationResult,
    VerificationVerdict,
    VerifierAgent,
)
from jarvis.core.enums import AgentRole
from jarvis.core.types import utc_now
from jarvis.memory.models import MemoryItem
from jarvis.memory.service import MemoryService
from jarvis.orchestrator.models import TaskRecord, TaskState, TaskStep
from jarvis.orchestrator.router import AgentRouter
from jarvis.orchestrator.store import TaskStore
from jarvis.providers.base import BaseLLMProvider, CompletionMessage
from jarvis.providers.registry import MultiModelRegistry
from jarvis.tools.base import ToolResult
from jarvis.tools.registry import ToolRegistry
from jarvis.utils.logging import setup_logger

logger = setup_logger(__name__)


class OrchestratorEngine:
    """
    Central workflow engine of A.O.R.I.S.
    Controls the entire lifecycle of user tasks from receipt to verified completion,
    orchestrating specialist models (OpenAI, Anthropic, Google) with NVIDIA Nemotron
    as the primary coordinator and final synthesis model.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        provider: BaseLLMProvider,
        tool_registry: ToolRegistry,
        task_store: TaskStore,
        model_registry: Optional[MultiModelRegistry] = None,
    ) -> None:
        self.memory_service = memory_service
        self.provider = provider
        self.tool_registry = tool_registry
        self.task_store = task_store
        self.model_registry = model_registry or MultiModelRegistry(
            settings=getattr(memory_service, "settings", None),
            orchestrator=provider,
        )

        # Specialized agents
        self.conversation_agent = ConversationAgent(provider)
        self.planner_agent = PlannerAgent(provider)
        self.research_agent = ResearchAgent(provider, tool_registry)
        self.coding_agent = CodingAgent(provider)
        self.computer_agent = ComputerAgent(provider, tool_registry)
        self.verifier_agent = VerifierAgent(provider)
        self.router = AgentRouter(provider)

    def initialize(self) -> None:
        """Initialize all backing services."""
        self.memory_service.initialize()
        self.task_store.initialize()
        logger.info("OrchestratorEngine initialized with MultiModelRegistry.")

    def close(self) -> None:
        """Safely close open resources."""
        self.memory_service.close()
        self.task_store.close()

    async def process_request(
        self,
        user_request: str,
        session_id: str = "default_session",
        user_confirmed: bool = False,
        specialist_override: Optional[str] = None,
    ) -> TaskRecord:
        """
        Process a new user request:
        1. Retrieve relevant memories
        2. Create and persist TaskRecord
        3. Route intent & select specialist model
        4. Execute agent / tool / specialist workflow
        5. Perform NVIDIA Nemotron final synthesis (if specialist was used)
        6. Verify results with VerifierAgent
        7. Ingest into persistent memory
        """
        # Step 1: Bounded context retrieval from memory
        relevant_memories: List[MemoryItem] = self.memory_service.retrieve_context(
            query=user_request, session_id=session_id
        )
        memory_ids = [m.id for m in relevant_memories]

        # Step 2: Create initial task record
        task = TaskRecord(
            session_id=session_id,
            user_request=user_request,
            state=TaskState.RUNNING,
            retrieved_memory_ids=memory_ids,
        )
        task.metadata["orchestrator"] = "NVIDIA Nemotron"
        task.metadata["orchestrator_model"] = self.provider.default_model
        self.task_store.save_task(task)

        # Step 3: Route request
        decision = self.router.route(user_request, specialist_override=specialist_override)
        task.workflow_type = decision.workflow_type
        task.intent = decision.intent
        task.complexity = decision.complexity
        task.capabilities = decision.required_capabilities
        task.metadata["confidence"] = decision.confidence
        task.metadata["rationale"] = decision.rationale
        task.metadata["specialist_routed"] = decision.specialist_model
        self.task_store.save_task(task)

        logger.info(
            f"Task {task.task_id} routed to '{decision.primary_agent.value}' "
            f"(Workflow: {task.workflow_type}, Intent: {task.intent}, Specialist: {decision.specialist_model})"
        )

        # Step 4: Execute workflow according to routing decision
        try:
            if decision.requires_planning:
                await self._execute_planned_workflow(task, relevant_memories, user_confirmed)
            elif decision.suggested_tool:
                await self._execute_direct_tool_workflow(
                    task, decision.suggested_tool, decision.tool_args, user_confirmed
                )
            else:
                await self._execute_agent_workflow(
                    task,
                    decision.primary_agent,
                    relevant_memories,
                    specialist_model=decision.specialist_model,
                    requires_synthesis=decision.requires_synthesis,
                )


        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}", exc_info=True)
            task.state = TaskState.FAILED
            task.error = str(e)
            task.final_response = f"I encountered an error executing your request: {e}"
            task.verification = VerificationResult(
                verdict=VerificationVerdict.FAILED,
                evidence=f"Execution raised exception: {e}",
                checks_performed=["exception_catcher"],
            )

        # Step 5: Post-execution ingestion into memory if completed or informative
        if task.state == TaskState.COMPLETED and task.final_response:
            self.memory_service.ingest_interaction(
                session_id=session_id,
                user_message=user_request,
                assistant_response=task.final_response,
            )

        task.updated_at = utc_now()
        if task.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            task.completed_at = utc_now()

        self.task_store.save_task(task)
        return task

    async def _execute_direct_tool_workflow(
        self,
        task: TaskRecord,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_confirmed: bool,
    ) -> None:
        """Execute a single approved tool directly."""
        step = TaskStep(
            step_id=1,
            description=f"Invoke tool '{tool_name}'",
            agent_role=AgentRole.COMPUTER,
            tool_name=tool_name,
            arguments=tool_args,
            status=TaskState.RUNNING,
        )
        task.plan = [step]

        tool_result = await self.computer_agent.execute_tool(
            tool_name=tool_name,
            arguments=tool_args,
            user_confirmed=user_confirmed,
        )

        step.result = tool_result.data
        step.error = tool_result.error

        if tool_result.requires_confirmation:
            step.status = TaskState.AWAITING_CONFIRMATION
            task.state = TaskState.AWAITING_CONFIRMATION
            task.pending_confirmation = {
                "step_id": step.step_id,
                "tool_name": tool_name,
                "arguments": tool_args,
                "prompt": tool_result.confirmation_prompt,
            }
            task.final_response = (
                f"ACTION REQUIRED: This operation requires your explicit confirmation.\n"
                f"{tool_result.confirmation_prompt}\n"
                f"Please approve or deny this task to proceed."
            )
            task.verification = VerificationResult(
                verdict=VerificationVerdict.INCOMPLETE,
                evidence="Waiting for user confirmation.",
                checks_performed=["confirmation_gate"],
            )
            return

        if not tool_result.success:
            step.status = TaskState.FAILED
            task.state = TaskState.FAILED
            task.error = tool_result.error
            task.final_response = f"Tool execution failed: {tool_result.error}"
            task.verification = VerificationResult(
                verdict=VerificationVerdict.FAILED,
                evidence=f"Tool failed: {tool_result.error}",
                checks_performed=["tool_execution_check"],
            )
            return

        step.status = TaskState.COMPLETED
        task.state = TaskState.COMPLETED

        # Format output message
        if tool_name == "calculator":
            task.final_response = f"Calculation Result: {tool_result.data.get('result')}"
        elif tool_name == "file_writer":
            task.final_response = f"Successfully wrote {tool_result.data.get('bytes_written')} bytes to {tool_result.data.get('path')}."
        elif tool_name == "file_reader":
            task.final_response = f"File Content ({tool_result.data.get('path')}):\n\n{tool_result.data.get('content')}"
        elif tool_name == "file_list":
            entries_str = ", ".join(e["name"] for e in tool_result.data.get("entries", []))
            task.final_response = f"Files in {tool_result.data.get('directory')}:\n{entries_str}"
        elif tool_name == "research_search":
            results = tool_result.data.get("results", [])
            citations = tool_result.data.get("citations", [])
            note = tool_result.data.get("note", "")
            if results:
                citations_str = "\n".join(f"- {c}" for c in citations)
                task.final_response = (
                    f"Research findings for '{tool_args.get('query')}':\n\n"
                    f"{note}\n\n"
                    f"Sources / Citations:\n{citations_str}"
                )
            else:
                task.final_response = f"Research inquiry '{tool_args.get('query')}': {note}"
        elif tool_name == "memory_search":
            memories = tool_result.data.get("memories", [])
            if memories:
                m_list = "\n".join(f"- [{m['category']}] {m['content']} (Confidence: {m['confidence']})" for m in memories)
                task.final_response = f"Found {len(memories)} matching memory records for '{tool_args.get('query')}':\n\n{m_list}"
            else:
                task.final_response = f"No memory records found matching query: '{tool_args.get('query')}'"
        elif tool_name == "pdf_generator":
            title = tool_result.data.get("title", "Document")
            file_name = tool_result.data.get("file_name", "output.pdf")
            bytes_written = tool_result.data.get("bytes_written", 0)
            dl_url = tool_result.data.get("download_url", "")
            task.final_response = (
                f"PDF Document Generated Successfully: '{title}'\n\n"
                f"File: {file_name} ({bytes_written} bytes)\n"
                f"Download Link: {dl_url}"
            )
            task.metadata["download_url"] = dl_url
            task.metadata["file_name"] = file_name
            task.metadata["bytes_written"] = bytes_written
        else:
            task.final_response = f"Result: {tool_result.data}"

        # Verify through VerifierAgent
        task.verification = self.verifier_agent.verify_deterministic(
            task_description=task.user_request,
            completion_criteria=f"Tool {tool_name} successfully executed",
            tool_results=[tool_result],
            context={},
        )

    async def _execute_agent_workflow(
        self,
        task: TaskRecord,
        role: AgentRole,
        memories: List[MemoryItem],
        specialist_model: Optional[str] = None,
        requires_synthesis: bool = False,
    ) -> None:
        """Route to a conversational, research, or coding agent with multi-model specialist and synthesis support."""
        agent_input = AgentInput(
            user_request=task.user_request,
            session_id=task.session_id,
            memories=memories,
        )

        # Check if specialist is requested/routed and available
        specialist_provider = self.model_registry.resolve_provider(specialist_model) if specialist_model else None
        used_specialist = specialist_provider is not None and specialist_provider != self.provider

        if used_specialist:
            logger.info(
                f"Task {task.task_id}: dispatching to specialist '{specialist_provider.provider_name}' "
                f"({specialist_provider.default_model})"
            )
            memory_context = "\n".join(f"- [{m.category.value}] {m.content}" for m in memories) if memories else "None"
            specialist_messages = [
                CompletionMessage(
                    role="system",
                    content=(
                        f"You are the {specialist_provider.provider_name} specialist working within A.O.R.I.S., "
                        "the multi-model AI assistant. Answer the user request with domain expertise.\n"
                        f"Known User Memory Context:\n{memory_context}"
                    ),
                ),
                CompletionMessage(role="user", content=task.user_request),
            ]
            try:
                spec_resp = await specialist_provider.generate_response(specialist_messages)
                specialist_content = spec_resp.content
                task.metadata["specialist_used"] = specialist_provider.provider_name
                task.metadata["specialist_model"] = specialist_provider.default_model
                task.metadata["specialist_tokens"] = spec_resp.usage

                if requires_synthesis:
                    logger.info(f"Task {task.task_id}: performing NVIDIA Nemotron final synthesis on specialist output.")
                    synth_messages = [
                        CompletionMessage(
                            role="system",
                            content=(
                                "You are A.O.R.I.S., an advanced multi-model AI assistant powered by NVIDIA Nemotron. "
                                "You are the primary orchestrator and final synthesis model. "
                                f"A specialist model ({specialist_provider.provider_name}) provided the following draft:\n\n"
                                f"<specialist_draft>\n{specialist_content}\n</specialist_draft>\n\n"
                                "Your task:\n"
                                "1. Review the specialist draft for factual accuracy, completeness, and clarity.\n"
                                "2. Verify that it directly satisfies the user's instructions.\n"
                                "3. Correct any discrepancies or tone inconsistencies.\n"
                                "4. Provide the final, authoritative response to the user."
                            ),
                        ),
                        CompletionMessage(
                            role="user",
                            content=f"Original Request: {task.user_request}\n\nPlease synthesize and finalize the response.",
                        ),
                    ]
                    synth_resp = await self.provider.generate_response(synth_messages)
                    task.final_response = synth_resp.content
                    task.metadata["synthesis_performed"] = True
                    task.metadata["orchestrator_model"] = self.provider.default_model
                else:
                    task.final_response = specialist_content
                    task.metadata["synthesis_performed"] = False

                task.state = TaskState.COMPLETED
                task.verification = self.verifier_agent.verify_deterministic(
                    task_description=task.user_request,
                    completion_criteria="Specialist execution and synthesis completed",
                    tool_results=[],
                    context={"specialist": specialist_provider.provider_name, "synthesis": requires_synthesis},
                )
                return
            except Exception as exc:
                logger.warning(
                    f"Specialist '{specialist_provider.provider_name}' execution failed: {exc}. "
                    "Falling back to primary orchestrator / standard agents."
                )
                task.metadata["specialist_fallback"] = True
                task.metadata["fallback_reason"] = str(exc)

        # Fallback or standard agent execution
        if specialist_model and not used_specialist:
            task.metadata["specialist_fallback"] = True
            task.metadata["fallback_reason"] = f"Specialist '{specialist_model}' credentials not configured or inactive."

        task.metadata["orchestrator_model"] = self.provider.default_model
        task.metadata["synthesis_performed"] = False

        if role == AgentRole.CODING:
            agent_output = await self.coding_agent.run(agent_input)
        elif role == AgentRole.RESEARCH:
            agent_output = await self.research_agent.run(agent_input)
        else:
            agent_output = await self.conversation_agent.run(agent_input)

        task.final_response = agent_output.response_text
        task.state = TaskState.COMPLETED
        task.verification = self.verifier_agent.verify_deterministic(
            task_description=task.user_request,
            completion_criteria="Respond helpfully",
            tool_results=[],
            context={"direct_answer": True},
        )


    async def _execute_planned_workflow(
        self,
        task: TaskRecord,
        memories: List[MemoryItem],
        user_confirmed: bool,
    ) -> None:
        """Decompose with PlannerAgent and execute steps sequentially with dependency piping and bounded replanning."""
        agent_input = AgentInput(
            user_request=task.user_request,
            session_id=task.session_id,
            memories=memories,
        )
        plan_output = await self.planner_agent.run(agent_input)
        steps_meta = plan_output.metadata.get("steps", [])

        task_steps: List[TaskStep] = []
        for s in steps_meta:
            task_steps.append(
                TaskStep(
                    step_id=s.get("step_id", 1),
                    description=s.get("description", "Execute plan step"),
                    agent_role=AgentRole(s.get("agent", "computer" if s.get("tool_name") else "conversation")),
                    tool_name=s.get("tool_name"),
                    arguments=s.get("arguments", {}),
                    requires_confirmation=s.get("requires_confirmation", False),
                    completion_criteria=s.get("completion_criteria"),
                )
            )
        task.plan = task_steps

        # Execute steps with output piping and bounded replanning
        tool_results: List[ToolResult] = []
        previous_step_result: Any = None
        max_recovery_attempts = 1
        recovery_attempts = 0

        idx = 0
        while idx < len(task.plan):
            step = task.plan[idx]
            task.current_step_index = idx
            step.status = TaskState.RUNNING

            # Dynamic argument substitution / output piping from previous step
            if step.arguments and previous_step_result is not None:
                for k, v in list(step.arguments.items()):
                    if isinstance(v, str):
                        if "{{previous_step_result}}" in v:
                            step.arguments[k] = v.replace("{{previous_step_result}}", str(previous_step_result))
                        elif "{{step_1_result}}" in v:
                            step.arguments[k] = v.replace("{{step_1_result}}", str(previous_step_result))

            if step.tool_name:
                tr = await self.computer_agent.execute_tool(
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                    user_confirmed=user_confirmed,
                )
                tool_results.append(tr)
                step.result = tr.data
                step.error = tr.error

                if tr.requires_confirmation:
                    step.status = TaskState.AWAITING_CONFIRMATION
                    task.state = TaskState.AWAITING_CONFIRMATION
                    task.pending_confirmation = {
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "arguments": step.arguments,
                        "prompt": tr.confirmation_prompt,
                    }
                    task.final_response = f"Plan paused at step {step.step_id}: requires confirmation.\n{tr.confirmation_prompt}"
                    task.verification = VerificationResult(
                        verdict=VerificationVerdict.INCOMPLETE,
                        evidence=f"Step {step.step_id} paused awaiting confirmation.",
                        checks_performed=["plan_confirmation_gate"],
                    )
                    return

                if not tr.success:
                    # Bounded Recovery / Replanning
                    if recovery_attempts < max_recovery_attempts:
                        recovery_attempts += 1
                        # Remove failed attempt from active tool_results as it is being superseded
                        if tool_results:
                            tool_results.pop()
                        logger.warning(
                            f"Step {step.step_id} failed: {tr.error}. Attempting replan {recovery_attempts}/{max_recovery_attempts}..."
                        )
                        task.replanned = True
                        completed_history = [s.model_dump() for s in task.plan[:idx] if s.status == TaskState.COMPLETED]
                        recovery_steps = await self.planner_agent.replan(
                            original_goal=task.user_request,
                            failed_step=step.model_dump(),
                            failure_reason=tr.error or "Unknown tool error",
                            completed_steps=completed_history,
                        )
                        task.replan_history.append({
                            "failed_step_id": step.step_id,
                            "error": tr.error,
                            "attempt": recovery_attempts,
                            "recovery_steps": recovery_steps,
                        })

                        if recovery_steps:
                            rec_step = recovery_steps[0]
                            step.description = f"[Replanned] {rec_step.get('description')}"
                            step.tool_name = rec_step.get("tool_name")
                            step.arguments = rec_step.get("arguments", {})
                            step.requires_confirmation = rec_step.get("requires_confirmation", False)
                            step.agent_role = AgentRole(rec_step.get("agent", "computer" if step.tool_name else "conversation"))
                            step.status = TaskState.PENDING
                            step.error = None
                            # Re-run current step with recovery step
                            continue

                    # If replan failed or recovery attempt exhausted
                    step.status = TaskState.FAILED
                    task.state = TaskState.FAILED
                    task.error = tr.error
                    task.final_response = f"Plan halted at step {step.step_id} ({step.description}): {tr.error}"
                    task.verification = self.verifier_agent.verify_deterministic(
                        task_description=task.user_request,
                        completion_criteria="Plan execution",
                        tool_results=tool_results,
                        context={"replanned": task.replanned},
                    )
                    return

                # Record successful step result for next steps
                if tr.data:
                    if "result" in tr.data:
                        previous_step_result = tr.data["result"]
                    elif "bytes_written" in tr.data:
                        previous_step_result = tr.data.get("path")
                    elif "content" in tr.data:
                        previous_step_result = tr.data.get("content")

            step.status = TaskState.COMPLETED
            idx += 1

        task.state = TaskState.COMPLETED
        # Format final plan summary
        summary_lines = [f"Completed multi-step workflow for: '{task.user_request}'"]
        for s in task.plan:
            status_mark = "[*]" if s.status == TaskState.COMPLETED else "[!]"
            res_str = f" -> {s.result}" if s.result else ""
            summary_lines.append(f"{status_mark} Step {s.step_id}: {s.description}{res_str}")
        if task.replanned:
            summary_lines.append("\n[RECOVERY]: Bounded replan was successfully executed to overcome an initial failure.")

        task.final_response = "\n".join(summary_lines)
        task.verification = self.verifier_agent.verify_deterministic(
            task_description=task.user_request,
            completion_criteria="Plan completion",
            tool_results=tool_results,
            context={"direct_answer": False, "replanned": task.replanned, "recovery_step_count": recovery_attempts},
        )

    async def approve_task(self, task_id: str) -> TaskRecord:
        """Approve and resume a task paused in awaiting_confirmation state."""
        task = self.task_store.get_task(task_id)
        if not task:
            raise ValueError(f"Task '{task_id}' not found.")
        if task.state != TaskState.AWAITING_CONFIRMATION or not task.pending_confirmation:
            return task

        conf = task.pending_confirmation
        tool_name = conf["tool_name"]
        arguments = conf["arguments"]
        step_id = conf.get("step_id")

        # Resume execution with user_confirmed=True
        tool_res = await self.computer_agent.execute_tool(
            tool_name=tool_name,
            arguments=arguments,
            user_confirmed=True,
        )

        task.pending_confirmation = None
        if tool_res.success:
            if task.plan and step_id:
                for step in task.plan:
                    if step.step_id == step_id:
                        step.status = TaskState.COMPLETED
                        step.result = tool_res.data
                        break
            task.state = TaskState.COMPLETED
            task.final_response = f"Confirmed and executed '{tool_name}' successfully: {tool_res.data}"
            task.verification = VerificationResult(
                verdict=VerificationVerdict.VERIFIED,
                evidence=f"Approved tool '{tool_name}' successfully completed.",
                checks_performed=["user_confirmation_received", "tool_execution"],
            )
        else:
            if task.plan and step_id:
                for step in task.plan:
                    if step.step_id == step_id:
                        step.status = TaskState.FAILED
                        step.error = tool_res.error
                        break
            task.state = TaskState.FAILED
            task.error = tool_res.error
            task.final_response = f"Approved tool '{tool_name}' failed: {tool_res.error}"
            task.verification = VerificationResult(
                verdict=VerificationVerdict.FAILED,
                evidence=f"Execution error: {tool_res.error}",
                checks_performed=["user_confirmation_received", "tool_error_detected"],
            )

        task.updated_at = utc_now()
        task.completed_at = utc_now()
        self.task_store.save_task(task)
        return task

    def deny_task(self, task_id: str, reason: str = "User denied execution") -> TaskRecord:
        """Deny and cancel a pending task."""
        task = self.task_store.get_task(task_id)
        if not task:
            raise ValueError(f"Task '{task_id}' not found.")

        task.state = TaskState.CANCELLED
        task.pending_confirmation = None
        task.error = reason
        task.final_response = f"Action cancelled: {reason}."
        task.verification = VerificationResult(
            verdict=VerificationVerdict.INCOMPLETE,
            evidence="Operation cancelled by user.",
            checks_performed=["user_cancellation"],
        )
        task.updated_at = utc_now()
        task.completed_at = utc_now()
        self.task_store.save_task(task)
        return task

    def cancel_task(self, task_id: str) -> TaskRecord:
        """Cancel a running or pending task."""
        return self.deny_task(task_id, reason="User cancelled task")

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Fetch task by ID."""
        return self.task_store.get_task(task_id)

    def list_tasks(self, session_id: Optional[str] = None, limit: int = 50) -> List[TaskRecord]:
        """List tasks."""
        return self.task_store.list_tasks(session_id=session_id, limit=limit)
