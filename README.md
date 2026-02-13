# Ultra Doc-Intelligence

AI-powered logistics document analysis backend with RAG, guardrails, and structured data extraction.

## Features

- **Document Upload**: Support for PDF, DOCX, and TXT files
- **RAG-based Q&A**: Ask questions about uploaded documents using retrieval-augmented generation
- **Structured Extraction**: Extract shipment data from logistics documents
- **Guardrails**: Configurable guardrails for response quality control
- **Vector Store**: Efficient document chunking and embedding storage

## Tech Stack

- **FastAPI** - Web framework
- **Groq** - LLM inference
- **[Sentence Transformers](https://huggingface.co/BAAI/bge-small-en-v1.5)** - Local document embeddings (model downloads automatically on first run, ~130MB)
- **PyMuPDF** - PDF parsing
- **python-docx** - DOCX parsing

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-skill-test-backend-local.git
   cd ai-skill-test-backend-local
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

5. Run the server:
   ```bash
   python main.py
   ```

   Or with uvicorn:
   ```bash
   uvicorn main:app --reload
   ```

The API will be available at `http://localhost:8000`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload a document |
| POST | `/ask` | Ask a question about a document |
| POST | `/extract` | Extract structured data |
| GET | `/documents` | List all documents |
| DELETE | `/documents/{doc_id}` | Delete a document |
| GET | `/guardrails` | Get guardrail settings |
| POST | `/guardrails/toggle` | Toggle a guardrail |
| POST | `/guardrails/add` | Add a custom guardrail |

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM inference | Yes |
| `GROQ_MODEL` | Model to use (default: configured in settings) | No |
| `EMBEDDING_MODEL` | Sentence transformer model | No |
| `MAX_FILE_SIZE_MB` | Maximum upload file size | No |


