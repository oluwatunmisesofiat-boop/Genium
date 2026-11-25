"""SQLite WAL discovery job queue and bounded background worker pool."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from discovery.models import DiscoverySeed
from registry.greenbook import normalize_nafdac_number
from registry.trusted_sources import default_genie_db_path

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str) -> str:
    return _SLUG_RE.sub("_", (text or "").strip().lower()).strip("_") or "product"


def normalize_target_key(seed: DiscoverySeed) -> str:
    """Build a deterministic deduplication key for a discovery seed."""
    if seed.nafdac_number:
        return f"nafdac:{normalize_nafdac_number(seed.nafdac_number)}"
    if seed.gtin:
        digits = re.sub(r"\D", "", seed.gtin)
        if digits:
            return f"gtin:{digits}"
    if seed.product_name:
        return f"name:{_slugify(seed.product_name)}"
    return f"anon:{uuid.uuid4().hex[:8]}"


def _row_to_job(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    for json_col, out_key in (("seed_json", "seed"), ("result_json", "result")):
        raw = data.pop(json_col, None)
        if raw:
            try:
                data[out_key] = json.loads(raw)
            except (TypeError, ValueError):
                data[out_key] = None
        else:
            data[out_key] = None
    return data


class JobStore:
    """Thread-safe SQLite WAL job registry inside `data/genie.db`."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_genie_db_path()
        self._lock = threading.Lock()
        self.init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS discovery_jobs (
                    job_id TEXT PRIMARY KEY,
                    target_key TEXT NOT NULL,
                    seed_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_discovery_jobs_target_status
                    ON discovery_jobs(target_key, status)
                """
            )

    def enqueue_or_get_active(self, seed: DiscoverySeed) -> Tuple[Dict[str, Any], bool]:
        """Atomically return an active (queued/running) job for this target, or create a new one.

        Returns `(job_dict, deduplicated)`.
        """
        target_key = normalize_target_key(seed)
        with self._lock:
            with self._connect() as conn:
                existing = conn.execute(
                    """
                    SELECT * FROM discovery_jobs
                    WHERE target_key = ? AND status IN ('queued', 'running')
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (target_key,),
                ).fetchone()
                if existing is not None:
                    return _row_to_job(existing), True

                job_id = f"disc_{uuid.uuid4().hex[:12]}"
                now = _utc_now()
                conn.execute(
                    """
                    INSERT INTO discovery_jobs (
                        job_id, target_key, seed_json, status, created_at
                    ) VALUES (?, ?, ?, 'queued', ?)
                    """,
                    (job_id, target_key, json.dumps(seed.to_dict(), ensure_ascii=False), now),
                )
                row = conn.execute(
                    "SELECT * FROM discovery_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                return _row_to_job(row), False

    def mark_running(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE discovery_jobs SET status = 'running', started_at = ? WHERE job_id = ?",
                (_utc_now(), job_id),
            )

    def mark_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE discovery_jobs
                SET status = 'completed', finished_at = ?, result_json = ?, error = NULL
                WHERE job_id = ?
                """,
                (_utc_now(), json.dumps(result, ensure_ascii=False), job_id),
            )

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE discovery_jobs
                SET status = 'failed', finished_at = ?, error = ?
                WHERE job_id = ?
                """,
                (_utc_now(), error[:1000], job_id),
            )

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM discovery_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _row_to_job(row) if row is not None else None

    def list_jobs(self, *, limit: int = 20, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM discovery_jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, max(1, limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM discovery_jobs ORDER BY created_at DESC LIMIT ?",
                    (max(1, limit)),
                ).fetchall()
        return [_row_to_job(r) for r in rows]


def default_discovery_worker(
    seed: DiscoverySeed,
    *,
    products_dir: Path,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run Step 2 (Discovery) + Step 3 (Synthesis + Evidence Gate) and write staged `product.json`."""
    from discovery import discover  # noqa: PLC0415
    from intelligence import LocalImageStorage, synthesize_dossier  # noqa: PLC0415
    from registry import TrustedSourceStore  # noqa: PLC0415
    from scripts.run_discovery import _db_record_payload, load_dotenv  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    trusted_store = TrustedSourceStore(db_path)
    trusted_store.ensure_seeded()

    bundle = discover(seed, trusted_store=trusted_store, max_urls=6)
    slug = _slugify(
        bundle.enriched_seed.product_name
        or bundle.enriched_seed.nafdac_number
        or seed.product_name
        or seed.nafdac_number
        or seed.gtin
        or "product"
    )
    product_dir = Path(products_dir) / slug
    product_dir.mkdir(parents=True, exist_ok=True)
    storage = LocalImageStorage(product_dir)

    dossier = synthesize_dossier(
        bundle,
        storage=storage,
        product_key="images",
        verify_visuals=True,
        trusted_store=trusted_store,
        trusted_only_images=True,
    )
    out_path = product_dir / "product.json"
    out_path.write_text(
        json.dumps(_db_record_payload(dossier), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "slug": slug,
        "product_json": str(out_path),
        "brand_name": dossier.brand_name,
        "nafdac_number": dossier.nafdac_number,
        "dossier_status": dossier.dossier_status,
        "reference_level": dossier.reference_level,
        "claims_count": len(dossier.claims),
        "elapsed_s": dossier.stats.get("elapsed_s"),
    }


class DiscoveryJobRunner:
    """Bounded background worker pool for deduplicated async discovery jobs."""

    def __init__(
        self,
        *,
        job_store: Optional[JobStore] = None,
        products_dir: Optional[Path | str] = None,
        max_workers: int = 2,
        worker_fn: Optional[Callable[[DiscoverySeed], Dict[str, Any]]] = None,
    ) -> None:
        self.job_store = job_store or JobStore()
        self.products_dir = (
            Path(products_dir)
            if products_dir is not None
            else (Path(__file__).resolve().parent.parent / "data" / "products")
        )
        self._worker_fn = worker_fn
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="genie-discovery",
        )
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, seed: DiscoverySeed) -> Tuple[Dict[str, Any], bool]:
        """Enqueue a discovery job (deduplicated against active queued/running jobs)."""
        job, deduplicated = self.job_store.enqueue_or_get_active(seed)
        if deduplicated:
            return job, True

        job_id = job["job_id"]
        fut = self._pool.submit(self._execute, job_id, seed)
        with self._lock:
            self._futures[job_id] = fut
        return job, False

    def _execute(self, job_id: str, seed: DiscoverySeed) -> None:
        self.job_store.mark_running(job_id)
        try:
            if self._worker_fn is not None:
                result = self._worker_fn(seed)
            else:
                result = default_discovery_worker(
                    seed,
                    products_dir=self.products_dir,
                    db_path=self.job_store.db_path,
                )
            self.job_store.mark_completed(job_id, result or {})
        except Exception as exc:  # noqa: BLE001
            self.job_store.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._futures.pop(job_id, None)

    def wait_for(self, job_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for an in-flight job to complete (useful in tests and CLI)."""
        with self._lock:
            fut = self._futures.get(job_id)
        if fut is not None:
            fut.result(timeout=timeout)
        return self.job_store.get_job(job_id)

    def active_count(self) -> int:
        with self._lock:
            return len(self._futures)

    def shutdown(self, *, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)
