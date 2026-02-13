"""
RAG pipeline module.
Handles retrieval-augmented generation using Groq LLM.
Includes retry logic, error classification, and confidence scoring.
"""

import json
import re
import time
import logging

from groq import Groq

from config import settings
from vector_store import retrieve_chunks
from guardrails import guardrail_engine
from exceptions import (
    LLMRateLimitError,
    LLMQuotaExceededError,
    LLMAuthError,
    LLMModelError,
    LLMContextLengthError,
    LLMAPIError,
    EmbeddingError,
)

logger = logging.getLogger(__name__)

MAX_LLM_RETRIES = 2
LLM_RETRY_BACKOFF = [5, 15]  # seconds to wait on retry 1, retry 2


def _get_groq_client() -> Groq:
    if not settings.GROQ_API_KEY:
        raise LLMAuthError("GROQ_API_KEY is not set. Add it to your .env file.")
    return Groq(api_key=settings.GROQ_API_KEY)


def _classify_and_raise(e: Exception, attempt: int, client: Groq, messages: list) -> str:
    """
    Classify a Groq API exception and either retry or raise a specific error.
    Returns response text if retry succeeds.
    """
    error_str = str(e).lower()
    error_full = str(e)

    # --- Rate limit (429) ---
    if "rate_limit" in error_str or "429" in error_str or "too many requests" in error_str:
        retry_after = 60
        if hasattr(e, 'response') and e.response is not None:
            retry_after = int(e.response.headers.get("retry-after", 60))

        if attempt < MAX_LLM_RETRIES:
            wait = min(max(retry_after, LLM_RETRY_BACKOFF[attempt]), 30)
            logger.warning(f"Rate limited. Waiting {wait}s before retry {attempt + 1}...")
            time.sleep(wait)
            return _call_groq_with_retry(client, messages, attempt + 1)

        raise LLMRateLimitError(
            f"Groq API rate limit exceeded. Please wait {retry_after} seconds and try again. "
            "Groq free tier allows ~30 requests/minute.",
            retry_after=retry_after,
        )

    # --- Quota / billing ---
    if "quota" in error_str or "insufficient" in error_str or "billing" in error_str or "credits" in error_str:
        raise LLMQuotaExceededError(
            "Groq API quota or credits exhausted. "
            "Check your account limits at https://console.groq.com. "
            "Free tier has daily token limits."
        )

    # --- Auth (401/403) ---
    if "auth" in error_str or "401" in error_str or "403" in error_str or "invalid api key" in error_str or "unauthorized" in error_str:
        raise LLMAuthError(
            "Groq API authentication failed. Your GROQ_API_KEY may be invalid or expired. "
            "Get a new key at https://console.groq.com."
        )

    # --- Model not found ---
    if "model" in error_str and ("not found" in error_str or "does not exist" in error_str or "unavailable" in error_str):
        raise LLMModelError(
            f"Model '{settings.GROQ_MODEL}' is not available on Groq. "
            "Check available models at https://console.groq.com/docs/models."
        )

    # --- Context length exceeded ---
    if "context" in error_str and ("length" in error_str or "too long" in error_str or "exceed" in error_str):
        raise LLMContextLengthError(
            "The document context is too long for the model's context window. "
            "Try uploading a shorter document or asking a more specific question."
        )

    if "token" in error_str and ("limit" in error_str or "exceed" in error_str or "maximum" in error_str):
        raise LLMContextLengthError(
            "Token limit exceeded. The document may be too large. "
            "Try a shorter document or more specific question."
        )

    # --- Server errors (500/502/503 - retryable) ---
    if "500" in error_str or "502" in error_str or "503" in error_str or "server" in error_str or "internal" in error_str:
        if attempt < MAX_LLM_RETRIES:
            logger.warning(f"Server error. Retrying in {LLM_RETRY_BACKOFF[attempt]}s...")
            time.sleep(LLM_RETRY_BACKOFF[attempt])
            return _call_groq_with_retry(client, messages, attempt + 1)
        raise LLMAPIError(
            f"Groq API server error after {MAX_LLM_RETRIES + 1} attempts. The service may be temporarily down."
        )

    # --- Timeout ---
    if "timeout" in error_str or "timed out" in error_str:
        if attempt < MAX_LLM_RETRIES:
            logger.warning(f"Timeout. Retrying in {LLM_RETRY_BACKOFF[attempt]}s...")
            time.sleep(LLM_RETRY_BACKOFF[attempt])
            return _call_groq_with_retry(client, messages, attempt + 1)
        raise LLMAPIError(
            "Groq API request timed out after retries. The service may be under heavy load — try again shortly."
        )

    # --- Connection errors ---
    if "connection" in error_str or "network" in error_str or "dns" in error_str:
        if attempt < MAX_LLM_RETRIES:
            time.sleep(LLM_RETRY_BACKOFF[attempt])
            return _call_groq_with_retry(client, messages, attempt + 1)
        raise LLMAPIError(
            "Cannot connect to Groq API. Check your internet connection and try again."
        )

    # --- Catch-all ---
    raise LLMAPIError(f"Unexpected Groq API error: {error_full}")


def _call_groq_with_retry(client: Groq, messages: list, attempt: int = 0) -> str:
    """
    Call Groq API with automatic retry for transient errors.
    Returns the raw response text.
    """
    try:
        chat_response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        return chat_response.choices[0].message.content

    except Exception as e:
        return _classify_and_raise(e, attempt, client, messages)


RAG_SYSTEM_PROMPT = """You are a precise document analysis assistant for logistics and transportation documents (Bill of Lading, Rate Confirmations, Shipment Instructions, Invoices, etc.).

LOGISTICS DOMAIN KNOWLEDGE (use this to interpret the document correctly):
- "Pickup" section = Shipper / Origin location (where goods are picked up FROM)
- "Drop" section = Consignee / Destination location (where goods are delivered TO)
- Lines marked [SHIPPER ...] or [CONSIGNEE ...] are explicit labels added during parsing
- "Shipper" and "Consignee" may appear as column headers in BOL documents
- The first address after "Shipper" is the shipper; the address after "Consignee" is the consignee
- "Carrier" = the trucking/transportation company moving the freight
- "Rate" or "Agreed Amount" = how much the carrier is being paid
- "Equipment" = trailer type (Flatbed, Dry Van, Reefer, etc.)
- "FTL" = Full Truckload, "LTL" = Less Than Truckload

STRICT RULES:
1. Answer ONLY from the provided document context below. Do NOT use any external knowledge.
2. If the answer is not found in the context, respond with exactly: "NOT_FOUND_IN_DOCUMENT"
3. Pay close attention to which entity is the SHIPPER vs CONSIGNEE — they are different.
4. Quote or closely paraphrase the relevant text from the context.
4. Be concise and direct.

After your answer, you MUST provide a JSON block on a new line in this exact format:
```json
{{"llm_confidence": 0.85, "grounded": true}}
```

- llm_confidence: float 0.0-1.0, your confidence that the answer is correct and fully supported by the context
- grounded: boolean, true if answer is fully supported by context, false otherwise

DOCUMENT CONTEXT:
{context}
"""


def _build_context_string(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks):
        pages = chunk["metadata"].get("pages", "unknown")
        score = chunk["similarity_score"]
        parts.append(
            f"[Source {i + 1} | Pages: {pages} | Relevance: {score:.2f}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _parse_llm_response(response_text: str) -> dict:
    """Parse LLM response to extract answer, confidence, and grounding flag."""
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{"llm_confidence".*?\})', response_text, re.DOTALL)

    llm_confidence = 0.5
    grounded = True

    if json_match:
        try:
            meta = json.loads(json_match.group(1))
            llm_confidence = float(meta.get("llm_confidence", 0.5))
            grounded = bool(meta.get("grounded", True))
        except (json.JSONDecodeError, ValueError):
            pass

    answer = response_text
    if json_match:
        answer = response_text[:json_match.start()].strip()
    answer = answer.rstrip("`").strip()

    if "NOT_FOUND_IN_DOCUMENT" in answer:
        grounded = False
        llm_confidence = 0.0
        answer = "Not found in document."

    return {"answer": answer, "llm_confidence": llm_confidence, "grounded": grounded}


def _compute_confidence(retrieval_scores: list[float], llm_confidence: float) -> float:
    """
    Blended confidence: 0.4 * avg_retrieval + 0.4 * llm_confidence + 0.2 * top_retrieval
    """
    if not retrieval_scores:
        return 0.0
    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
    top_retrieval = max(retrieval_scores)
    blended = (0.4 * avg_retrieval) + (0.4 * llm_confidence) + (0.2 * top_retrieval)
    return round(min(max(blended, 0.0), 1.0), 4)


def ask_question(question: str, doc_id: str, custom_guardrail_context: dict = None) -> dict:
    """
    Full RAG pipeline: retrieve -> guardrails -> generate -> score -> validate.
    Returns structured response with answer, confidence, sources, guardrail results.
    Raises AppError subclasses for API/infrastructure failures.
    """
    # Step 1: Retrieve relevant chunks
    try:
        retrieved = retrieve_chunks(query=question, doc_id=doc_id)
    except Exception as e:
        raise EmbeddingError(f"Failed to retrieve document chunks: {str(e)}")

    # Step 2: Pre-generation guardrail checks
    pre_context = {
        "retrieved_chunks": retrieved,
        "question": question,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        **(custom_guardrail_context or {}),
    }

    from guardrails import check_empty_context, check_retrieval_similarity

    empty_check = check_empty_context(pre_context)
    similarity_check = check_retrieval_similarity(pre_context)

    if not empty_check.passed:
        return {
            "answer": empty_check.message,
            "confidence_score": 0.0,
            "sources": [],
            "guardrail_results": [_guardrail_to_dict(empty_check), _guardrail_to_dict(similarity_check)],
            "all_guardrails_passed": False,
            "error": None,
        }

    if not similarity_check.passed:
        return {
            "answer": similarity_check.message,
            "confidence_score": 0.0,
            "sources": _format_sources(retrieved),
            "guardrail_results": [_guardrail_to_dict(empty_check), _guardrail_to_dict(similarity_check)],
            "all_guardrails_passed": False,
            "error": None,
        }

    # Step 3: Generate answer via Groq LLM (with retry)
    context_str = _build_context_string(retrieved)
    client = _get_groq_client()

    raw_response = _call_groq_with_retry(
        client=client,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=context_str)},
            {"role": "user", "content": question},
        ],
    )

    parsed = _parse_llm_response(raw_response)

    # Step 4: Compute confidence
    retrieval_scores = [c["similarity_score"] for c in retrieved]
    confidence = _compute_confidence(retrieval_scores, parsed["llm_confidence"])

    # Step 5: Run all guardrails (post-generation)
    full_context = {
        "retrieved_chunks": retrieved,
        "question": question,
        "answer": parsed["answer"],
        "confidence_score": confidence,
        "llm_grounded": parsed["grounded"],
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "confidence_threshold": settings.CONFIDENCE_LOW_THRESHOLD,
        **(custom_guardrail_context or {}),
    }

    guardrail_results = guardrail_engine.run_guardrails(full_context)
    all_passed = guardrail_engine.all_passed(guardrail_results)

    final_answer = parsed["answer"]
    if not all_passed:
        failed = [r for r in guardrail_results if not r.passed]
        warnings = "; ".join(f"[{r.name}] {r.message}" for r in failed)
        final_answer = f"⚠️ Guardrail Warning: {warnings}\n\n{parsed['answer']}"

    return {
        "answer": final_answer,
        "confidence_score": confidence,
        "sources": _format_sources(retrieved),
        "guardrail_results": [_guardrail_to_dict(r) for r in guardrail_results],
        "all_guardrails_passed": all_passed,
        "error": None,
    }


def _format_sources(chunks: list[dict]) -> list[dict]:
    return [
        {
            "text": c["text"][:500],
            "pages": c["metadata"].get("pages", "unknown"),
            "similarity_score": c["similarity_score"],
            "chunk_index": c["metadata"].get("chunk_index", -1),
        }
        for c in chunks
    ]


def _guardrail_to_dict(result) -> dict:
    return {
        "name": result.name,
        "passed": result.passed,
        "message": result.message,
        "is_default": result.is_default,
    }