import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
    CONFIDENCE_LOW_THRESHOLD: float = float(os.getenv("CONFIDENCE_LOW_THRESHOLD", "0.4"))
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))


settings = Settings()
