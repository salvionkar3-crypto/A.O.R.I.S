"""Primitive types, helpers, and base models for JARVIS."""

from datetime import datetime, timezone
from typing import Any, Dict
import uuid
from pydantic import BaseModel, ConfigDict


def generate_id(prefix: str = "") -> str:
    """Generate a unique random UUID string, optionally prefixed."""
    val = uuid.uuid4().hex[:16]
    return f"{prefix}_{val}" if prefix else val


def utc_now() -> datetime:
    """Return the current datetime in UTC."""
    return datetime.now(timezone.utc)


class JarvisBaseModel(BaseModel):
    """Base Pydantic model with strict validation and standard configuration."""

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
