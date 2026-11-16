"""Domain enumeration types for JARVIS."""

from enum import Enum


class MemoryType(str, Enum):
    """The four conceptual memory dimensions."""
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class FactStatus(str, Enum):
    """Distinguishes the certainty and provenance of stored information."""
    CONFIRMED_FACT = "confirmed_fact"      # Empirically verified or ground truth
    USER_CLAIM = "user_claim"              # Directly stated by the user
    SYSTEM_INFERENCE = "system_inference"  # Extracted/inferred by the agent/model


class MemoryCategory(str, Enum):
    """Semantic categorization of memory items."""
    PREFERENCE = "preference"  # User tastes, style preferences, habits
    PROJECT = "project"        # Project details, architectures, conventions
    ENTITY = "entity"          # People, services, organizations, devices
    RULE = "rule"              # Operating constraints, instructions, prohibitions
    WORKFLOW = "workflow"      # Reusable multi-step procedure
    GENERAL = "general"        # General world knowledge or unclassified facts


class Role(str, Enum):
    """Participant role in conversation turns and episodes."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class TaskStatus(str, Enum):
    """Execution status for multi-agent tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class AgentRole(str, Enum):
    """Specialized agents in the multi-agent orchestration architecture."""
    ROUTER = "router"
    PLANNER = "planner"
    RESEARCH = "research"
    CODING = "coding"
    COMPUTER = "computer"
    CONVERSATION = "conversation"
    VERIFIER = "verifier"
