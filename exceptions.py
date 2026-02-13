"""
Custom exception classes for Ultra Doc-Intelligence.
Provides specific error types for different failure modes.
"""


class AppError(Exception):
    """Base application error with user-friendly message and error type."""
    def __init__(self, message: str, error_type: str = "error", retry_after: int = None, status_code: int = 500):
        self.message = message
        self.error_type = error_type
        self.retry_after = retry_after
        self.status_code = status_code
        super().__init__(self.message)


class LLMRateLimitError(AppError):
    """Groq API rate limit (429) exceeded."""
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message=message, error_type="rate_limit_exceeded", retry_after=retry_after, status_code=429)


class LLMQuotaExceededError(AppError):
    """Groq API quota/credits exhausted."""
    def __init__(self, message: str):
        super().__init__(message=message, error_type="quota_exceeded", status_code=402)


class LLMAuthError(AppError):
    """Groq API authentication failure."""
    def __init__(self, message: str):
        super().__init__(message=message, error_type="auth_error", status_code=401)


class LLMModelError(AppError):
    """Groq model not found or unavailable."""
    def __init__(self, message: str):
        super().__init__(message=message, error_type="model_error", status_code=400)


class LLMContextLengthError(AppError):
    """Document context exceeds model's token limit."""
    def __init__(self, message: str):
        super().__init__(message=message, error_type="context_length_exceeded", status_code=413)


class LLMAPIError(AppError):
    """Generic Groq API error."""
    def __init__(self, message: str):
        super().__init__(message=message, error_type="llm_api_error", status_code=502)


class EmbeddingError(AppError):
    """Embedding model failure."""
    def __init__(self, message: str):
        super().__init__(message=message, error_type="embedding_error", status_code=500)


class VectorStoreError(AppError):
    """ChromaDB / vector store failure."""
    def __init__(self, message: str):
        super().__init__(message=message, error_type="vector_store_error", status_code=500)


class DocumentParseError(AppError):
    """Failed to parse uploaded document."""
    def __init__(self, message: str):
        super().__init__(message=message, error_type="document_parse_error", status_code=400)
