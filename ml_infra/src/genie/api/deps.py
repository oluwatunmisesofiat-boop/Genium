"""Dependency injection for FastAPI stateless ML endpoints."""
from genie.storage.vector_store import InMemoryVectorStore, VectorStore

# In-memory vector store singleton for local runtime (swap with QdrantVectorStore in prod)
_vector_store_instance = InMemoryVectorStore()


def get_vector_store() -> VectorStore:
    return _vector_store_instance
