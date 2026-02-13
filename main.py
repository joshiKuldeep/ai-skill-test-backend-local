"""
Ultra Doc-Intelligence - FastAPI Backend
A RAG-based logistics document Q&A system with error handling.
"""

import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from config import settings
from document_parser import parse_document
from chunker import chunk_text
from vector_store import store_chunks, document_exists, delete_document
from rag import ask_question
from extractor import extract_structured_data
from guardrails import guardrail_engine
from exceptions import AppError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- In-memory document store (POC only) ---
document_store: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load embedding model on startup."""
    from embeddings import get_model
    logger.info("Loading embedding model...")
    try:
        get_model()
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
    yield


app = FastAPI(
    title="Ultra Doc-Intelligence",
    description="AI-powered logistics document analysis with RAG, guardrails, and structured extraction.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================
# Global Exception Handler
# ============================

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """
    Catch all AppError subclasses and return structured JSON errors.
    Frontend can parse error_type to show appropriate UI messages.
    """
    response_data = {
        "detail": exc.message,
        "error_type": exc.error_type,
    }
    if exc.retry_after is not None:
        response_data["retry_after"] = exc.retry_after

    return JSONResponse(status_code=exc.status_code, content=response_data)


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Catch unexpected errors and return a safe message."""
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected internal error occurred. Please try again.",
            "error_type": "internal_error",
        },
    )


# ============================
# Request / Response Models
# ============================

class AskRequest(BaseModel):
    doc_id: str
    question: str
    custom_guardrail_context: Optional[dict] = None


class ExtractRequest(BaseModel):
    doc_id: str


class GuardrailToggleRequest(BaseModel):
    name: str
    enabled: bool


class CustomGuardrailRequest(BaseModel):
    name: str
    description: str
    guardrail_type: str
    params: dict = Field(default_factory=dict)


# ============================
# Endpoints
# ============================

@app.get("/health")
async def health_check():
    """Health check with configuration info."""
    api_key_set = bool(settings.GROQ_API_KEY)
    return {
        "status": "healthy" if api_key_set else "degraded",
        "model": settings.GROQ_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "groq_api_key_set": api_key_set,
        "warnings": [] if api_key_set else ["GROQ_API_KEY is not set — /ask and /extract will fail"],
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (PDF, DOCX, TXT)."""

    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".txt"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail={
            "message": f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}",
            "error_type": "invalid_file_type",
        })

    # Read file
    file_bytes = await file.read()

    # Validate size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail={
            "message": f"File too large: {size_mb:.1f}MB. Maximum: {settings.MAX_FILE_SIZE_MB}MB",
            "error_type": "file_too_large",
        })

    # Parse
    try:
        raw_text = parse_document(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"message": str(e), "error_type": "parse_error"})
    except Exception as e:
        logger.error(f"Document parse failed: {e}")
        raise HTTPException(status_code=500, detail={
            "message": f"Failed to parse document: {str(e)}",
            "error_type": "parse_error",
        })

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail={
            "message": "Document appears empty or text could not be extracted. For scanned PDFs, OCR is not yet supported.",
            "error_type": "empty_document",
        })

    # Chunk & embed
    doc_id = str(uuid.uuid4())[:12]
    try:
        chunks = chunk_text(raw_text, doc_id=doc_id)
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise HTTPException(status_code=500, detail={
            "message": f"Failed to chunk document: {str(e)}",
            "error_type": "chunking_error",
        })

    if not chunks:
        raise HTTPException(status_code=400, detail={
            "message": "Failed to create any text chunks from the document.",
            "error_type": "chunking_error",
        })

    try:
        num_stored = store_chunks(chunks, doc_id)
    except Exception as e:
        logger.error(f"Vector store failed: {e}")
        raise HTTPException(status_code=500, detail={
            "message": f"Failed to store document embeddings: {str(e)}. The embedding model may not have loaded correctly.",
            "error_type": "embedding_error",
        })

    # Store raw text
    document_store[doc_id] = {
        "filename": file.filename,
        "raw_text": raw_text,
        "num_chunks": num_stored,
        "text_length": len(raw_text),
    }

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "num_chunks": num_stored,
        "text_length": len(raw_text),
        "message": f"Document processed successfully. {num_stored} chunks indexed.",
    }


@app.post("/ask")
async def ask(request: AskRequest):
    """
    Ask a question about an uploaded document.
    Raises structured errors for rate limits, auth failures, etc.
    """
    if request.doc_id not in document_store:
        raise HTTPException(status_code=404, detail={
            "message": f"Document '{request.doc_id}' not found. Please upload a document first.",
            "error_type": "document_not_found",
        })

    if not request.question.strip():
        raise HTTPException(status_code=400, detail={
            "message": "Question cannot be empty.",
            "error_type": "empty_question",
        })

    # AppError subclasses (rate limit, auth, etc.) are caught by the global handler
    result = ask_question(
        question=request.question,
        doc_id=request.doc_id,
        custom_guardrail_context=request.custom_guardrail_context,
    )

    return result


@app.post("/extract")
async def extract(request: ExtractRequest):
    """
    Extract structured shipment data.
    Raises structured errors for rate limits, auth failures, etc.
    """
    if request.doc_id not in document_store:
        raise HTTPException(status_code=404, detail={
            "message": f"Document '{request.doc_id}' not found. Please upload a document first.",
            "error_type": "document_not_found",
        })

    raw_text = document_store[request.doc_id]["raw_text"]

    # AppError subclasses caught by global handler
    extracted = extract_structured_data(raw_text)

    return {
        "doc_id": request.doc_id,
        "filename": document_store[request.doc_id]["filename"],
        "extracted_data": extracted,
    }


# --- Guardrail Endpoints ---

@app.get("/guardrails")
async def get_guardrails():
    return {"guardrails": guardrail_engine.get_all_guardrails()}


@app.post("/guardrails/toggle")
async def toggle_guardrail(request: GuardrailToggleRequest):
    if request.enabled:
        success = guardrail_engine.enable_custom_guardrail(request.name)
    else:
        success = guardrail_engine.disable_custom_guardrail(request.name)

    if not success:
        raise HTTPException(status_code=404, detail={
            "message": f"Custom guardrail '{request.name}' not found. Default guardrails cannot be toggled.",
            "error_type": "guardrail_not_found",
        })
    return {"message": f"Guardrail '{request.name}' {'enabled' if request.enabled else 'disabled'}.", "guardrails": guardrail_engine.get_all_guardrails()}


@app.post("/guardrails/add")
async def add_guardrail(request: CustomGuardrailRequest):
    success = guardrail_engine.add_custom_guardrail(
        name=request.name,
        description=request.description,
        guardrail_type=request.guardrail_type,
        params=request.params,
    )
    if not success:
        raise HTTPException(status_code=400, detail={
            "message": f"Invalid guardrail_type: {request.guardrail_type}. Options: min_similarity, max_answer_length, min_chunks",
            "error_type": "invalid_guardrail_type",
        })
    return {"message": f"Custom guardrail '{request.name}' added.", "guardrails": guardrail_engine.get_all_guardrails()}


@app.get("/documents")
async def list_documents():
    docs = []
    for doc_id, info in document_store.items():
        docs.append({
            "doc_id": doc_id,
            "filename": info["filename"],
            "num_chunks": info["num_chunks"],
            "text_length": info["text_length"],
        })
    return {"documents": docs}


@app.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str):
    if doc_id not in document_store:
        raise HTTPException(status_code=404, detail={
            "message": f"Document '{doc_id}' not found.",
            "error_type": "document_not_found",
        })
    delete_document(doc_id)
    del document_store[doc_id]
    return {"message": f"Document {doc_id} deleted."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
