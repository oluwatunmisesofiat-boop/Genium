"""Vector store interface, in-memory implementation, and SQLite + NumPy cosine similarity store."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import aiosqlite
import numpy as np


@dataclass
class VectorSearchResult:
    id: str
    score: float
    metadata: Dict[str, Any]


class VectorStore(Protocol):
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        ...

    async def upsert(
        self,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        ...

    async def count(self) -> int:
        ...


class InMemoryVectorStore:
    """In-memory cosine similarity vector store supporting metadata list and exact filtering."""

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []

    async def upsert(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        normalized_v = v / (norm + 1e-9)

        # Update if exists
        for entry in self._entries:
            if entry["id"] == id:
                entry["vector"] = normalized_v
                entry["metadata"] = metadata
                return

        self._entries.append({
            "id": id,
            "vector": normalized_v,
            "metadata": metadata,
        })

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        if not self._entries:
            return []

        qv = np.array(query_vector, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-9)

        results = []
        for entry in self._entries:
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    entry_val = entry["metadata"].get(k)
                    if isinstance(entry_val, list):
                        if v not in entry_val:
                            match = False
                            break
                    else:
                        if entry_val != v:
                            match = False
                            break
                if not match:
                    continue

            score = float(np.dot(qv, entry["vector"]))
            results.append(
                VectorSearchResult(
                    id=entry["id"],
                    score=score,
                    metadata=entry["metadata"],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def count(self) -> int:
        return len(self._entries)


class SQLiteVectorStore:
    """SQLite-backed vector store using aiosqlite and NumPy cosine similarity."""

    def __init__(self, db_path: Path | str = Path("./data/vectors.db")):
        self.db_path = Path(db_path)
        self._initialized = False

    async def _ensure_table(self, db: aiosqlite.Connection) -> None:
        if not self._initialized:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS pack_embeddings (
                    id TEXT PRIMARY KEY,
                    vector_blob BLOB NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            await db.commit()
            self._initialized = True

    @staticmethod
    def _matches_filter(metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        for k, v in filter_metadata.items():
            entry_val = metadata.get(k)
            if isinstance(entry_val, list):
                if v not in entry_val:
                    return False
            elif entry_val != v:
                return False
        return True

    async def upsert(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        v = np.array(vector, dtype=np.float32)
        normalized_v = v / (np.linalg.norm(v) + 1e-9)
        blob = normalized_v.tobytes()
        meta_json = json.dumps(metadata)

        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table(db)
            await db.execute(
                """
                INSERT INTO pack_embeddings (id, vector_blob, metadata_json)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    vector_blob = excluded.vector_blob,
                    metadata_json = excluded.metadata_json
                """,
                (id, blob, meta_json),
            )
            await db.commit()

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        if not self.db_path.exists():
            return []

        qv = np.array(query_vector, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-9)

        results: List[VectorSearchResult] = []
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table(db)
            async with db.execute(
                "SELECT id, vector_blob, metadata_json FROM pack_embeddings"
            ) as cursor:
                async for row_id, blob, meta_json in cursor:
                    metadata = json.loads(meta_json)
                    if filter_metadata and not self._matches_filter(metadata, filter_metadata):
                        continue
                    vec = np.frombuffer(blob, dtype=np.float32)
                    score = float(np.dot(qv, vec))
                    results.append(
                        VectorSearchResult(id=row_id, score=score, metadata=metadata)
                    )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def count(self) -> int:
        if not self.db_path.exists():
            return 0
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table(db)
            async with db.execute("SELECT COUNT(*) FROM pack_embeddings") as cursor:
                row = await cursor.fetchone()
                return int(row[0]) if row else 0
