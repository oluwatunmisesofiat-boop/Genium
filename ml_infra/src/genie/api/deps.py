"""Dependency injection for FastAPI stateless ML endpoints."""
from genie.common.config import settings
from genie.storage.vector_store import SQLiteVectorStore, VectorStore

# SQLite + NumPy cosine similarity vector store singleton for local/hackathon runtime
_vector_store_instance = SQLiteVectorStore(db_path=settings.VECTOR_DB_PATH)


def get_vector_store() -> VectorStore:
    return _vector_store_instance
