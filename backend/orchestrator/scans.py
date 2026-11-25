"""SQLite WAL + JSON artifact persistence for every GENIE AI verification scan run."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ScanStore:
    """Persists every verification run to SQLite (`verification_scans`) and `data/scans/<scan_id>.json`."""

    def __init__(self, db_path: Path | str, scans_dir: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.scans_dir = Path(scans_dir) if scans_dir is not None else (self.db_path.parent / "scans")
        self.scans_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_scans (
                    scan_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    source_endpoint TEXT NOT NULL,
                    product_name TEXT,
                    nafdac_number TEXT,
                    verdict TEXT NOT NULL,
                    ml_infra_verdict TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    coverage REAL NOT NULL,
                    reference_level TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    limitations_json TEXT NOT NULL DEFAULT '[]',
                    counterfeit_tell_hits_json TEXT NOT NULL DEFAULT '[]',
                    discovery_job_id TEXT,
                    execution_time_ms REAL NOT NULL,
                    json_path TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            # Safe idempotent migration for existing SQLite databases
            existing_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(verification_scans)").fetchall()
            }
            for col_name, col_def in (
                ("summary", "TEXT NOT NULL DEFAULT ''"),
                ("limitations_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("counterfeit_tell_hits_json", "TEXT NOT NULL DEFAULT '[]'"),
            ):
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE verification_scans ADD COLUMN {col_name} {col_def}"
                    )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scans_created_at ON verification_scans(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scans_nafdac ON verification_scans(nafdac_number)"
            )

    @staticmethod
    def new_scan_id() -> str:
        return f"scan_{uuid.uuid4().hex[:12]}"

    def record_scan(
        self,
        *,
        scan_id: str,
        source_endpoint: str,
        product_name: Optional[str],
        nafdac_number: Optional[str],
        verdict: str,
        ml_infra_verdict: str,
        confidence: float,
        coverage: float,
        reference_level: str,
        summary: str = "",
        limitations: Optional[List[str]] = None,
        counterfeit_tell_hits: Optional[List[Dict[str, Any]]] = None,
        discovery_job_id: Optional[str],
        execution_time_ms: float,
        payload: Dict[str, Any],
    ) -> str:
        """Persist scan metadata + summary/limitations + full payload to SQLite WAL and JSON."""
        created_at = _utc_now_iso()
        json_file = self.scans_dir / f"{scan_id}.json"
        lims = list(limitations or [])
        hits = list(counterfeit_tell_hits or [])

        enriched_payload = dict(payload)
        enriched_payload["scan_id"] = scan_id
        enriched_payload["scan_json_path"] = str(json_file)
        enriched_payload["created_at"] = created_at
        enriched_payload["source_endpoint"] = source_endpoint
        enriched_payload["summary"] = summary
        enriched_payload["limitations"] = lims
        enriched_payload["counterfeit_tell_hits"] = hits

        serialized = json.dumps(enriched_payload, indent=2, default=str)
        tmp_file = json_file.with_suffix(".json.tmp")
        tmp_file.write_text(serialized, encoding="utf-8")
        tmp_file.replace(json_file)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO verification_scans (
                    scan_id, created_at, source_endpoint, product_name, nafdac_number,
                    verdict, ml_infra_verdict, confidence, coverage, reference_level,
                    summary, limitations_json, counterfeit_tell_hits_json,
                    discovery_job_id, execution_time_ms, json_path, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan_id,
                    created_at,
                    source_endpoint,
                    product_name,
                    nafdac_number,
                    verdict,
                    ml_infra_verdict,
                    float(confidence),
                    float(coverage),
                    reference_level,
                    summary,
                    json.dumps(lims),
                    json.dumps(hits, default=str),
                    discovery_job_id,
                    float(execution_time_ms),
                    str(json_file),
                    serialized,
                ),
            )
        return str(json_file)

    def update_scan_payload(
        self,
        scan_id: str,
        payload: Dict[str, Any],
        *,
        source_endpoint: Optional[str] = None,
    ) -> str:
        """Update the saved JSON file and SQLite `payload_json` for an existing scan_id."""
        json_file = self.scans_dir / f"{scan_id}.json"
        enriched_payload = dict(payload)
        enriched_payload["scan_id"] = scan_id
        enriched_payload["scan_json_path"] = str(json_file)
        if source_endpoint:
            enriched_payload["source_endpoint"] = source_endpoint

        ver_res = (enriched_payload.get("verification") or {}).get("result") or enriched_payload.get("result") or {}
        if isinstance(ver_res, dict):
            enriched_payload.setdefault("summary", ver_res.get("summary") or "")
            enriched_payload.setdefault("limitations", ver_res.get("limitations") or [])
            enriched_payload.setdefault(
                "counterfeit_tell_hits", ver_res.get("counterfeit_tell_hits") or []
            )

        serialized = json.dumps(enriched_payload, indent=2, default=str)
        tmp_file = json_file.with_suffix(".json.tmp")
        tmp_file.write_text(serialized, encoding="utf-8")
        tmp_file.replace(json_file)

        with self._connect() as conn:
            if source_endpoint:
                conn.execute(
                    "UPDATE verification_scans SET source_endpoint = ?, payload_json = ?, json_path = ? WHERE scan_id = ?",
                    (source_endpoint, serialized, str(json_file), scan_id),
                )
            else:
                conn.execute(
                    "UPDATE verification_scans SET payload_json = ?, json_path = ? WHERE scan_id = ?",
                    (serialized, str(json_file), scan_id),
                )
        return str(json_file)

    def get_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM verification_scans WHERE scan_id = ?",
                (scan_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row, include_payload=True)

    def list_scans(
        self,
        *,
        limit: int = 20,
        verdict: Optional[str] = None,
        nafdac_number: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if verdict:
            clauses.append("verdict = ?")
            params.append(verdict)
        if nafdac_number:
            clauses.append("nafdac_number = ?")
            params.append(nafdac_number)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(100, int(limit))))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM verification_scans {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_dict(r, include_payload=False) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM verification_scans").fetchone()
        return int(row["n"]) if row else 0

    @staticmethod
    def _row_to_dict(row: sqlite3.Row, *, include_payload: bool) -> Dict[str, Any]:
        try:
            lims = json.loads(row["limitations_json"] or "[]")
        except Exception:  # noqa: BLE001
            lims = []
        try:
            hits = json.loads(row["counterfeit_tell_hits_json"] or "[]")
        except Exception:  # noqa: BLE001
            hits = []

        out: Dict[str, Any] = {
            "scan_id": row["scan_id"],
            "created_at": row["created_at"],
            "source_endpoint": row["source_endpoint"],
            "product_name": row["product_name"],
            "nafdac_number": row["nafdac_number"],
            "verdict": row["verdict"],
            "ml_infra_verdict": row["ml_infra_verdict"],
            "confidence": row["confidence"],
            "coverage": row["coverage"],
            "reference_level": row["reference_level"],
            "summary": row["summary"],
            "limitations": lims,
            "counterfeit_tell_hits": hits,
            "discovery_job_id": row["discovery_job_id"],
            "execution_time_ms": row["execution_time_ms"],
            "json_path": row["json_path"],
        }
        if include_payload:
            try:
                out["payload"] = json.loads(row["payload_json"])
            except Exception:  # noqa: BLE001
                out["payload"] = {}
        return out
