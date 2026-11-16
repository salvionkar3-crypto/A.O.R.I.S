"""Domain exceptions for JARVIS."""


class JarvisError(Exception):
    """Base exception for all JARVIS errors."""
    pass


class MemoryError(JarvisError):
    """Base exception for memory operations."""
    pass


class StorageError(MemoryError):
    """Raised when low-level persistent storage operations fail."""
    pass


class MemoryNotFoundError(MemoryError):
    """Raised when an expected memory item cannot be found."""
    pass


class ConflictError(MemoryError):
    """Raised when a direct contradiction cannot be automatically resolved."""
    pass


class ProviderError(JarvisError):
    """Raised when an external model provider fails or is unreachable."""
    pass


class AuthenticationError(ProviderError):
    """Raised when provider credentials (API key) are invalid or missing."""
    pass


class RateLimitError(ProviderError):
    """Raised when provider rate limits are exceeded (HTTP 429)."""
    pass


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""
    pass


class ProviderUnavailableError(ProviderError):
    """Raised when the provider endpoint is down or unreachable (5xx/connection error)."""
    pass


class MalformedResponseError(ProviderError):
    """Raised when provider returns an unparseable or schema-violating response."""
    pass


class ToolExecutionError(JarvisError):
    """Raised when a tool execution encounters an unrecoverable failure."""
    pass


class ToolPermissionError(ToolExecutionError):
    """Raised when a tool execution violates permission or confirmation requirements."""
    pass


class ToolValidationError(ToolExecutionError):
    """Raised when tool arguments fail validation or path traversal is detected."""
    pass
