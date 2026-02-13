"""
Guardrails module.
Provides default (non-editable) and custom (user-defined) guardrails.
Each guardrail is a function that takes context and returns (passed: bool, message: str).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from config import settings


@dataclass
class GuardrailResult:
    name: str
    passed: bool
    message: str
    is_default: bool = True


@dataclass
class Guardrail:
    name: str
    description: str
    check_fn: Callable
    is_default: bool = True
    enabled: bool = True


# --- Default Guardrail Functions ---

def check_retrieval_similarity(context: dict) -> GuardrailResult:
    """
    DEFAULT: Reject if top retrieval similarity is below threshold.
    Ensures answers are grounded in actual document content.
    """
    chunks = context.get("retrieved_chunks", [])
    threshold = context.get("similarity_threshold", settings.SIMILARITY_THRESHOLD)

    if not chunks:
        return GuardrailResult(
            name="Retrieval Similarity Gate",
            passed=False,
            message="No relevant content found in the document.",
            is_default=True,
        )

    top_score = chunks[0]["similarity_score"]
    if top_score < threshold:
        return GuardrailResult(
            name="Retrieval Similarity Gate",
            passed=False,
            message=f"Not found in document. Best match similarity ({top_score:.2f}) is below threshold ({threshold:.2f}).",
            is_default=True,
        )

    return GuardrailResult(
        name="Retrieval Similarity Gate",
        passed=True,
        message=f"Top similarity: {top_score:.2f} (threshold: {threshold:.2f})",
        is_default=True,
    )


def check_confidence_threshold(context: dict) -> GuardrailResult:
    """
    DEFAULT: Warn or reject if blended confidence score is too low.
    """
    confidence = context.get("confidence_score", 0)
    threshold = context.get("confidence_threshold", settings.CONFIDENCE_LOW_THRESHOLD)

    if confidence < threshold:
        return GuardrailResult(
            name="Confidence Threshold",
            passed=False,
            message=f"Low confidence ({confidence:.2f}). The answer may not be reliable. Threshold: {threshold:.2f}.",
            is_default=True,
        )

    return GuardrailResult(
        name="Confidence Threshold",
        passed=True,
        message=f"Confidence: {confidence:.2f} (threshold: {threshold:.2f})",
        is_default=True,
    )


def check_empty_context(context: dict) -> GuardrailResult:
    """
    DEFAULT: Reject if no context chunks were retrieved at all.
    """
    chunks = context.get("retrieved_chunks", [])

    if not chunks:
        return GuardrailResult(
            name="Empty Context Check",
            passed=False,
            message="No document context available to answer this question.",
            is_default=True,
        )

    return GuardrailResult(
        name="Empty Context Check",
        passed=True,
        message=f"Retrieved {len(chunks)} context chunks.",
        is_default=True,
    )


def check_answer_grounding(context: dict) -> GuardrailResult:
    """
    DEFAULT: Verify the LLM indicated the answer is grounded.
    Checks the LLM's self-reported grounding flag.
    """
    llm_grounded = context.get("llm_grounded", True)

    if not llm_grounded:
        return GuardrailResult(
            name="Answer Grounding Check",
            passed=False,
            message="The model indicated the answer could not be fully grounded in the document.",
            is_default=True,
        )

    return GuardrailResult(
        name="Answer Grounding Check",
        passed=True,
        message="Answer is grounded in document content.",
        is_default=True,
    )


# --- Custom Guardrail Functions ---

def check_min_chunk_agreement(context: dict) -> GuardrailResult:
    """
    CUSTOM: Require at least N chunks above similarity threshold.
    Ensures answer has multiple supporting sources.
    """
    chunks = context.get("retrieved_chunks", [])
    threshold = context.get("similarity_threshold", settings.SIMILARITY_THRESHOLD)
    min_chunks = context.get("min_agreeing_chunks", 2)

    agreeing = [c for c in chunks if c["similarity_score"] >= threshold]

    if len(agreeing) < min_chunks:
        return GuardrailResult(
            name="Chunk Agreement",
            passed=False,
            message=f"Only {len(agreeing)} chunks agree (need {min_chunks}). Answer may be weakly supported.",
            is_default=False,
        )

    return GuardrailResult(
        name="Chunk Agreement",
        passed=True,
        message=f"{len(agreeing)} chunks agree above threshold.",
        is_default=False,
    )


def check_max_answer_length(context: dict) -> GuardrailResult:
    """
    CUSTOM: Reject overly long answers that may indicate hallucination.
    """
    answer = context.get("answer", "")
    max_words = context.get("max_answer_words", 200)
    word_count = len(answer.split())

    if word_count > max_words:
        return GuardrailResult(
            name="Answer Length Check",
            passed=False,
            message=f"Answer is {word_count} words (max: {max_words}). May contain hallucinated content.",
            is_default=False,
        )

    return GuardrailResult(
        name="Answer Length Check",
        passed=True,
        message=f"Answer length: {word_count} words (max: {max_words}).",
        is_default=False,
    )


def check_question_relevance(context: dict) -> GuardrailResult:
    """
    CUSTOM: Check if the question seems related to logistics/shipping.
    """
    question = context.get("question", "").lower()
    chunks = context.get("retrieved_chunks", [])

    # If retrieval returned something relevant, the question is fine
    if chunks and chunks[0]["similarity_score"] >= settings.SIMILARITY_THRESHOLD:
        return GuardrailResult(
            name="Question Relevance",
            passed=True,
            message="Question is relevant to document content.",
            is_default=False,
        )

    # Check for completely off-topic questions
    off_topic_indicators = ["recipe", "weather", "movie", "song", "joke", "poem"]
    if any(word in question for word in off_topic_indicators):
        return GuardrailResult(
            name="Question Relevance",
            passed=False,
            message="Question does not appear related to the uploaded document.",
            is_default=False,
        )

    return GuardrailResult(
        name="Question Relevance",
        passed=True,
        message="Question relevance check passed.",
        is_default=False,
    )


# --- Guardrail Registry ---

DEFAULT_GUARDRAILS = [
    Guardrail(
        name="Retrieval Similarity Gate",
        description="Reject answer if top retrieval similarity is below the configured threshold.",
        check_fn=check_retrieval_similarity,
        is_default=True,
    ),
    Guardrail(
        name="Confidence Threshold",
        description="Reject answer if blended confidence score is too low.",
        check_fn=check_confidence_threshold,
        is_default=True,
    ),
    Guardrail(
        name="Empty Context Check",
        description="Reject if no context chunks were retrieved from the document.",
        check_fn=check_empty_context,
        is_default=True,
    ),
    Guardrail(
        name="Answer Grounding Check",
        description="Verify the LLM reports the answer as grounded in the document.",
        check_fn=check_answer_grounding,
        is_default=True,
    ),
]

AVAILABLE_CUSTOM_GUARDRAILS = [
    Guardrail(
        name="Chunk Agreement",
        description="Require at least N chunks to agree above similarity threshold.",
        check_fn=check_min_chunk_agreement,
        is_default=False,
        enabled=False,
    ),
    Guardrail(
        name="Answer Length Check",
        description="Reject answers that exceed a maximum word count (may indicate hallucination).",
        check_fn=check_max_answer_length,
        is_default=False,
        enabled=False,
    ),
    Guardrail(
        name="Question Relevance",
        description="Check if the question is related to the document content.",
        check_fn=check_question_relevance,
        is_default=False,
        enabled=False,
    ),
]


class GuardrailEngine:
    """Manages and executes guardrails."""

    def __init__(self):
        self.default_guardrails = list(DEFAULT_GUARDRAILS)
        self.custom_guardrails = list(AVAILABLE_CUSTOM_GUARDRAILS)

    def get_all_guardrails(self) -> list[dict]:
        """Return all guardrails with their status."""
        result = []
        for g in self.default_guardrails:
            result.append({
                "name": g.name,
                "description": g.description,
                "is_default": True,
                "enabled": True,  # defaults are always enabled
                "editable": False,
            })
        for g in self.custom_guardrails:
            result.append({
                "name": g.name,
                "description": g.description,
                "is_default": False,
                "enabled": g.enabled,
                "editable": True,
            })
        return result

    def enable_custom_guardrail(self, name: str) -> bool:
        """Enable a custom guardrail by name."""
        for g in self.custom_guardrails:
            if g.name == name:
                g.enabled = True
                return True
        return False

    def disable_custom_guardrail(self, name: str) -> bool:
        """Disable a custom guardrail by name."""
        for g in self.custom_guardrails:
            if g.name == name:
                g.enabled = False
                return True
        return False

    def add_custom_guardrail(
        self,
        name: str,
        description: str,
        guardrail_type: str,
        params: dict = None,
    ) -> bool:
        """
        Add a new user-defined guardrail.
        guardrail_type: 'min_similarity' | 'max_answer_length' | 'min_chunks'
        """
        params = params or {}

        if guardrail_type == "min_similarity":
            threshold = params.get("threshold", 0.5)

            def check_fn(context):
                chunks = context.get("retrieved_chunks", [])
                if not chunks or chunks[0]["similarity_score"] < threshold:
                    return GuardrailResult(
                        name=name,
                        passed=False,
                        message=f"Similarity below custom threshold ({threshold}).",
                        is_default=False,
                    )
                return GuardrailResult(name=name, passed=True, message="Passed.", is_default=False)

        elif guardrail_type == "max_answer_length":
            max_words = params.get("max_words", 150)

            def check_fn(context):
                answer = context.get("answer", "")
                wc = len(answer.split())
                if wc > max_words:
                    return GuardrailResult(
                        name=name,
                        passed=False,
                        message=f"Answer too long: {wc} words (max {max_words}).",
                        is_default=False,
                    )
                return GuardrailResult(name=name, passed=True, message="Passed.", is_default=False)

        elif guardrail_type == "min_chunks":
            min_count = params.get("min_count", 2)

            def check_fn(context):
                chunks = context.get("retrieved_chunks", [])
                threshold = context.get("similarity_threshold", settings.SIMILARITY_THRESHOLD)
                agreeing = [c for c in chunks if c["similarity_score"] >= threshold]
                if len(agreeing) < min_count:
                    return GuardrailResult(
                        name=name,
                        passed=False,
                        message=f"Only {len(agreeing)} agreeing chunks (need {min_count}).",
                        is_default=False,
                    )
                return GuardrailResult(name=name, passed=True, message="Passed.", is_default=False)

        else:
            return False

        self.custom_guardrails.append(
            Guardrail(
                name=name,
                description=description,
                check_fn=check_fn,
                is_default=False,
                enabled=True,
            )
        )
        return True

    def run_guardrails(self, context: dict) -> list[GuardrailResult]:
        """
        Execute all enabled guardrails against the given context.
        Returns list of results.
        """
        results = []

        # Always run default guardrails
        for g in self.default_guardrails:
            result = g.check_fn(context)
            results.append(result)

        # Run enabled custom guardrails
        for g in self.custom_guardrails:
            if g.enabled:
                result = g.check_fn(context)
                results.append(result)

        return results

    def all_passed(self, results: list[GuardrailResult]) -> bool:
        """Check if all guardrail results passed."""
        return all(r.passed for r in results)


# Global instance
guardrail_engine = GuardrailEngine()
