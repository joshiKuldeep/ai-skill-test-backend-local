"""
Structured extraction module.
Extracts shipment data fields from document text using Groq LLM.
Includes error handling for API failures.
"""

import json
import re
import time
import logging

from groq import Groq

from config import settings
from exceptions import (
    LLMRateLimitError,
    LLMQuotaExceededError,
    LLMAuthError,
    LLMModelError,
    LLMContextLengthError,
    LLMAPIError,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF = [5, 15]


EXTRACTION_PROMPT = """You are a logistics document data extractor. Extract structured shipment data from the provided document text.

IMPORTANT LOGISTICS CONTEXT:
- "Pickup" section = Shipper / Origin (where goods are picked up FROM)
- "Drop" section = Consignee / Destination (where goods are delivered TO)
- Lines marked [SHIPPER ...] are the shipper/origin location
- Lines marked [CONSIGNEE ...] are the consignee/destination location
- "Load ID" or "Reference ID" = shipment_id
- "Agreed Amount" = rate
- Data may appear in [STRUCTURED DATA EXTRACTED FROM TABLES] blocks — prefer these as they are more accurate
- The same field may appear multiple times — use the most specific/complete value

Extract the following fields. If a field is not found in the document, set it to null.

Fields to extract:
- shipment_id: Shipment, load, or reference identifier (e.g., "LD53657")
- shipper: Shipper/origin name and address (from Pickup section or [SHIPPER ...] label)
- consignee: Consignee/destination name and address (from Drop section or [CONSIGNEE ...] label)
- pickup_datetime: Pickup/shipping date and time (ISO 8601 if possible)
- delivery_datetime: Delivery date and time (ISO 8601 if possible)
- equipment_type: Equipment/trailer type (e.g., Flatbed, Dry Van, Reefer)
- mode: Transportation mode (e.g., FTL, LTL, Intermodal)
- rate: Rate/cost amount (numeric value only, e.g., 400.00)
- currency: Currency code (e.g., USD, CAD)
- weight: Weight with unit (e.g., "56000 lbs")
- carrier_name: Carrier company name

RESPOND WITH ONLY A VALID JSON OBJECT. No explanation, no markdown fences, no extra text.

Example output:
{{"shipment_id": "LD53657", "shipper": "AAA, Los Angeles International Airport (LAX), World Way, Los Angeles, CA, USA", "consignee": "xyz, 7470 Cherry Avenue, Fontana, CA 92336, USA", "pickup_datetime": "2026-02-08T09:00:00", "delivery_datetime": "2026-02-08T09:00:00", "equipment_type": "Flatbed", "mode": "FTL", "rate": 400.00, "currency": "USD", "weight": "56000 lbs", "carrier_name": "SWIFT SHIFT LOGISTICS LLC"}}

DOCUMENT TEXT:
{document_text}
"""

EXPECTED_FIELDS = [
    "shipment_id", "shipper", "consignee", "pickup_datetime", "delivery_datetime",
    "equipment_type", "mode", "rate", "currency", "weight", "carrier_name",
]


def _classify_and_raise(e: Exception, attempt: int, client: Groq, messages: list) -> str:
    """Classify Groq error and retry or raise."""
    error_str = str(e).lower()
    error_full = str(e)

    if "rate_limit" in error_str or "429" in error_str or "too many requests" in error_str:
        retry_after = 60
        if hasattr(e, 'response') and e.response is not None:
            retry_after = int(e.response.headers.get("retry-after", 60))
        if attempt < MAX_RETRIES:
            wait = min(max(retry_after, RETRY_BACKOFF[attempt]), 30)
            logger.warning(f"Rate limited during extraction. Waiting {wait}s...")
            time.sleep(wait)
            return _call_with_retry(client, messages, attempt + 1)
        raise LLMRateLimitError(
            f"Rate limit exceeded during extraction. Wait {retry_after}s and try again.",
            retry_after=retry_after,
        )

    if "quota" in error_str or "insufficient" in error_str or "billing" in error_str:
        raise LLMQuotaExceededError(
            "Groq API quota exhausted. Check your account at https://console.groq.com."
        )

    if "auth" in error_str or "401" in error_str or "403" in error_str or "invalid api key" in error_str:
        raise LLMAuthError(
            "Groq API authentication failed. Check your GROQ_API_KEY."
        )

    if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
        raise LLMModelError(f"Model '{settings.GROQ_MODEL}' is not available on Groq.")

    if ("context" in error_str or "token" in error_str) and ("length" in error_str or "exceed" in error_str or "limit" in error_str):
        raise LLMContextLengthError(
            "Document is too long for extraction. Try a shorter document."
        )

    if "500" in error_str or "502" in error_str or "503" in error_str or "timeout" in error_str or "connection" in error_str:
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF[attempt])
            return _call_with_retry(client, messages, attempt + 1)
        raise LLMAPIError(f"Groq API error during extraction after retries: {error_full}")

    raise LLMAPIError(f"Unexpected error during extraction: {error_full}")


def _call_with_retry(client: Groq, messages: list, attempt: int = 0) -> str:
    """Call Groq with retry logic."""
    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _classify_and_raise(e, attempt, client, messages)


def extract_structured_data(document_text: str) -> dict:
    """
    Extract structured shipment data from document text.
    Returns dict with all expected fields (null if missing).
    Raises AppError subclasses on API failures.
    """
    if not settings.GROQ_API_KEY:
        raise LLMAuthError("GROQ_API_KEY is not set. Add it to your .env file.")

    client = Groq(api_key=settings.GROQ_API_KEY)

    # Truncate if too long (keep first ~6000 words)
    words = document_text.split()
    if len(words) > 6000:
        document_text = " ".join(words[:6000])

    messages = [
        {"role": "system", "content": "You are a precise data extraction assistant. Output ONLY valid JSON."},
        {"role": "user", "content": EXTRACTION_PROMPT.format(document_text=document_text)},
    ]

    raw = _call_with_retry(client, messages)

    # Clean up response
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                extracted = json.loads(json_match.group())
            except json.JSONDecodeError:
                extracted = {}
        else:
            extracted = {}

    # Ensure all expected fields present
    result = {}
    for field in EXPECTED_FIELDS:
        result[field] = extracted.get(field, None)

    return result