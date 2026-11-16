"""Sanitized and structured logging for JARVIS."""

import logging
import re
import sys
from typing import List, Tuple

# Patterns to match and their respective replacements
SENSITIVE_REPLACEMENTS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(r"(?i)(api[_-]?key|token|secret|password|bearer\s+)[:=]\s*([^\s,;]+)"),
        r"\1: [REDACTED]",
    ),
    (
        re.compile(r"nvapi-[A-Za-z0-9_-]+"),
        "[REDACTED_API_KEY]",
    ),
    (
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        "[REDACTED_SECRET]",
    ),
]


class SecretRedactionFilter(logging.Filter):
    """Logging filter that redacts detected secrets and API keys directly on records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                for pattern, replacement in SENSITIVE_REPLACEMENTS:
                    record.msg = pattern.sub(replacement, record.msg)
            if record.args:
                if isinstance(record.args, tuple):
                    record.args = tuple(self._sanitize_val(arg) for arg in record.args)
                elif isinstance(record.args, dict):
                    record.args = {k: self._sanitize_val(v) for k, v in record.args.items()}
        except Exception:
            pass
        return True

    def _sanitize_val(self, val):
        if isinstance(val, str):
            for pattern, replacement in SENSITIVE_REPLACEMENTS:
                val = pattern.sub(replacement, val)
        return val


class SecretRedactingFormatter(logging.Formatter):
    """Logging formatter that redacts detected secrets and API keys."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        redacted = original
        for pattern, replacement in SENSITIVE_REPLACEMENTS:
            redacted = pattern.sub(replacement, redacted)
        return redacted


def setup_logger(name: str = "jarvis", level: str = "INFO") -> logging.Logger:
    """Configure and return a sanitized application logger."""
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Attach secret redacting filter to logger if not present
    if not any(isinstance(f, SecretRedactionFilter) for f in logger.filters):
        logger.addFilter(SecretRedactionFilter())

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = SecretRedactingFormatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

