"""NAFDAC Greenbook local SQLite + FTS5 mirror and AJAX sync client.

Mirrors the official NAFDAC Registered Drug Database (greenbook.nafdac.gov.ng)
into a local SQLite database for millisecond lookups during both Discovery
and Verification without relying on live portal availability.
"""
from __future__ import annotations

import html
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

import httpx

DEFAULT_GREENBOOK_URL = "https://greenbook.nafdac.gov.ng"
DEFAULT_HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; GenieAI/0.1)",
}
_NAFDAC_QUERY_RE = re.compile(r"^(?:[A-Z]\d|\d{2})\s*[-–—]\s*\d{4,6}[A-Z]?$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_DASH_WS_RE = re.compile(r"\s*[-–—]\s*")
_NON_NAME_MARKERS_RE = re.compile(r"[#*]+")


def default_db_path() -> Path:
    """Return the configured Greenbook SQLite path (GREENBOOK_DB_PATH or data/greenbook.db)."""
    env_path = os.environ.get("GREENBOOK_DB_PATH", "").strip()
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent.parent / "data" / "greenbook.db"


def normalize_nafdac_number(raw: object) -> str:
    """Normalize whitespace, dashes, HTML entities, and casing on a NAFDAC code."""
    text = html.unescape(str(raw or "")).strip().upper()
    return _DASH_WS_RE.sub("-", text)


def _clean_text(value: object, *, multiline: bool = False) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value)).replace("\r\n", "\n").replace("\r", "\n")
    if multiline:
        lines = [" ".join(line.split()) for line in text.split("\n")]
        return "\n".join(line for line in lines if line).strip()
    return " ".join(text.split())


def _nested_str(raw: Dict, top_key: str, nested_key: str, sub_key: str) -> str:
    val = raw.get(top_key)
    if isinstance(val, str) and val.strip():
        return _clean_text(val)
    nested = raw.get(nested_key)
    if isinstance(nested, dict):
        sub_val = nested.get(sub_key)
        if sub_val is not None:
            return _clean_text(sub_val)
    return ""


def normalize_row(raw: Dict, *, synced_at: Optional[str] = None) -> Dict:
    """Normalize one raw Greenbook DataTables JSON row into our SQLite schema."""
    product_id = int(raw["product_id"])
    raw_name = str(raw.get("product_name") or "")
    clean_name = _clean_text(_NON_NAME_MARKERS_RE.sub("", raw_name))
    nafdac = normalize_nafdac_number(raw.get("NAFDAC") or raw.get("nafdac_number"))

    smpc = str(raw.get("smpc") or raw.get("smpc_url") or "").strip()
    if not smpc.startswith(("http://", "https://")):
        smpc = ""

    return {
        "product_id": product_id,
        "nafdac_number": nafdac,
        "product_name": clean_name,
        "raw_product_name": raw_name,
        "ingredient_name": _nested_str(raw, "ingredient_name", "ingredient", "ingredient_name"),
        "ingredient_synonym": _nested_str(raw, "synonym", "ingredient", "synonym")
        or _clean_text(raw.get("ingredient_synonym")),
        "category_name": _nested_str(raw, "category_name", "product_category", "name"),
        "form_name": _nested_str(raw, "form_name", "form", "name"),
        "route_name": _nested_str(raw, "route_name", "route", "name"),
        "strength": _clean_text(raw.get("strength")),
        "applicant_name": _nested_str(raw, "applicant_name", "applicant", "name"),
        "approval_date": _clean_text(raw.get("approval_date")),
        "expiry_date": _clean_text(raw.get("expiry_date")),
        "status": _clean_text(raw.get("status")),
        "product_description": _clean_text(raw.get("product_description"), multiline=True),
        "pack_size": _clean_text(raw.get("pack_size")),
        "composition": _clean_text(raw.get("composition"), multiline=True),
        "smpc_url": smpc,
        "atc_code": _clean_text(raw.get("atc") or raw.get("atc_code")),
        "detail_url": _clean_text(raw.get("detail_url"))
        or f"{DEFAULT_GREENBOOK_URL}/products/details/{product_id}",
        "updated_at": _clean_text(raw.get("updated_at")),
        "synced_at": synced_at or datetime.now(timezone.utc).isoformat(),
    }


_COLUMNS = (
    "product_id",
    "nafdac_number",
    "product_name",
    "raw_product_name",
    "ingredient_name",
    "ingredient_synonym",
    "category_name",
    "form_name",
    "route_name",
    "strength",
    "applicant_name",
    "approval_date",
    "expiry_date",
    "status",
    "product_description",
    "pack_size",
    "composition",
    "smpc_url",
    "atc_code",
    "detail_url",
    "updated_at",
    "synced_at",
)


class GreenbookStore:
    """SQLite + FTS5 repository for NAFDAC Greenbook records."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self._fts_available: Optional[bool] = None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            with conn:
                yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create tables, indexes, and FTS5 index if they do not exist."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS greenbook_products (
                    product_id INTEGER PRIMARY KEY,
                    nafdac_number TEXT NOT NULL,
                    product_name TEXT NOT NULL,
                    raw_product_name TEXT,
                    ingredient_name TEXT,
                    ingredient_synonym TEXT,
                    category_name TEXT,
                    form_name TEXT,
                    route_name TEXT,
                    strength TEXT,
                    applicant_name TEXT,
                    approval_date TEXT,
                    expiry_date TEXT,
                    status TEXT,
                    product_description TEXT,
                    pack_size TEXT,
                    composition TEXT,
                    smpc_url TEXT,
                    atc_code TEXT,
                    detail_url TEXT,
                    updated_at TEXT,
                    synced_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_greenbook_nafdac
                    ON greenbook_products(nafdac_number);
                CREATE INDEX IF NOT EXISTS idx_greenbook_name
                    ON greenbook_products(product_name);
                CREATE INDEX IF NOT EXISTS idx_greenbook_applicant
                    ON greenbook_products(applicant_name);
                CREATE INDEX IF NOT EXISTS idx_greenbook_status
                    ON greenbook_products(status);

                CREATE TABLE IF NOT EXISTS sync_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS greenbook_fts USING fts5(
                        product_name,
                        ingredient_name,
                        ingredient_synonym,
                        applicant_name,
                        nafdac_number,
                        strength,
                        composition,
                        content='greenbook_products',
                        content_rowid='product_id'
                    );
                    """
                )
                self._fts_available = True
            except sqlite3.OperationalError:
                self._fts_available = False

    def _has_fts(self, conn: sqlite3.Connection) -> bool:
        if self._fts_available is not None:
            return self._fts_available
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='greenbook_fts'"
        ).fetchone()
        self._fts_available = row is not None
        return self._fts_available

    def rebuild_fts(self) -> None:
        """Rebuild the FTS5 index from greenbook_products."""
        with self._connect() as conn:
            if self._has_fts(conn):
                conn.execute("INSERT INTO greenbook_fts(greenbook_fts) VALUES('rebuild');")

    def upsert_rows(self, rows: Iterable[Dict], *, rebuild_fts: bool = True) -> int:
        """Insert or update Greenbook rows; accepts raw AJAX dicts or normalized dicts."""
        self.init_schema()
        now_iso = datetime.now(timezone.utc).isoformat()
        prepared: List[tuple] = []
        for raw in rows:
            if "product_id" not in raw:
                continue
            row = raw if "nafdac_number" in raw and "detail_url" in raw else normalize_row(raw, synced_at=now_iso)
            prepared.append(tuple(row.get(col, "") for col in _COLUMNS))
        if not prepared:
            return 0

        placeholders = ", ".join(["?"] * len(_COLUMNS))
        updates = ", ".join(f"{col}=excluded.{col}" for col in _COLUMNS if col != "product_id")
        sql = (
            f"INSERT INTO greenbook_products ({', '.join(_COLUMNS)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT(product_id) DO UPDATE SET {updates};"
        )
        with self._connect() as conn:
            conn.executemany(sql, prepared)
            if rebuild_fts and self._has_fts(conn):
                conn.execute("INSERT INTO greenbook_fts(greenbook_fts) VALUES('rebuild');")
        return len(prepared)

    def sync_from_remote(
        self,
        *,
        base_url: str = DEFAULT_GREENBOOK_URL,
        batch_size: int = 500,
        max_records: Optional[int] = None,
        delay_s: float = 0.2,
        timeout: float = 45.0,
        progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
        fetcher: Optional[Callable[[int, int], Dict]] = None,
    ) -> Dict[str, object]:
        """Paginate the Greenbook AJAX DataTables endpoint and mirror all rows into SQLite."""
        self.init_schema()
        start = 0
        draw = 1
        fetched = 0
        total: Optional[int] = None

        def _default_fetch(client: httpx.Client, offset: int, length: int, draw_num: int) -> Dict:
            params = {
                "draw": draw_num,
                "start": offset,
                "length": length,
                "columns[0][data]": "product_name",
                "columns[0][name]": "product_name",
                "columns[0][searchable]": "true",
                "columns[0][orderable]": "true",
                "columns[0][search][value]": "",
            }
            last_exc: Optional[Exception] = None
            for attempt in range(3):
                try:
                    resp = client.get(base_url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
                    resp.raise_for_status()
                    return resp.json()
                except (httpx.HTTPError, ValueError) as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(2**attempt)
            raise RuntimeError(f"Greenbook sync failed at offset={offset}: {last_exc}") from last_exc

        client = None if fetcher is not None else httpx.Client(follow_redirects=True)
        try:
            while True:
                if max_records is not None and fetched >= max_records:
                    break
                req_len = batch_size
                if max_records is not None:
                    req_len = min(batch_size, max_records - fetched)

                if fetcher is not None:
                    payload = fetcher(start, req_len)
                else:
                    assert client is not None
                    payload = _default_fetch(client, start, req_len, draw)

                if total is None and payload.get("recordsTotal") is not None:
                    total = int(payload["recordsTotal"])

                batch = payload.get("data") or []
                if not batch:
                    break
                if max_records is not None and fetched + len(batch) > max_records:
                    batch = batch[: max_records - fetched]

                count = self.upsert_rows(batch, rebuild_fts=False)
                fetched += count
                start += len(batch)
                draw += 1

                if progress_cb is not None:
                    progress_cb(fetched, total)

                if total is not None and start >= total:
                    break
                if max_records is not None and fetched >= max_records:
                    break
                if delay_s > 0:
                    time.sleep(delay_s)
        finally:
            if client is not None:
                client.close()

        self.rebuild_fts()
        synced_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO sync_meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                [
                    ("last_synced_at", synced_at),
                    ("records_total", str(total if total is not None else self.count())),
                    ("records_stored", str(self.count())),
                ],
            )
        return {
            "synced_at": synced_at,
            "fetched": fetched,
            "records_total": total,
            "records_stored": self.count(),
            "db_path": str(self.db_path),
        }

    def lookup_by_nafdac(self, nafdac_number: str) -> List[Dict]:
        """Exact normalized lookup by NAFDAC registration number."""
        code = normalize_nafdac_number(nafdac_number)
        if not code or not self.db_path.exists():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM greenbook_products WHERE UPPER(nafdac_number) = ? ORDER BY product_id ASC;",
                (code,),
            ).fetchall()
        return [dict(r) for r in rows]

    def search(
        self,
        query: str,
        *,
        manufacturer: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Search mirrored Greenbook records by product name, ingredient, applicant, or NAFDAC code."""
        cleaned = _clean_text(query)
        if not cleaned or not self.db_path.exists():
            return []

        if _NAFDAC_QUERY_RE.match(cleaned):
            exact = self.lookup_by_nafdac(cleaned)
            if exact:
                return _prioritize_manufacturer(exact, manufacturer)[:limit]

        tokens = [t for t in _TOKEN_RE.findall(cleaned) if len(t) >= 2]
        if not tokens:
            return []

        fetch_limit = max(limit * 3, 25)
        rows: List[Dict] = []

        with self._connect() as conn:
            if self._has_fts(conn):
                and_expr = " AND ".join(f'"{t}"*' for t in tokens)
                rows = _fts_query(conn, and_expr, fetch_limit)
                if not rows and len(tokens) > 1:
                    or_expr = " OR ".join(f'"{t}"*' for t in tokens)
                    rows = _fts_query(conn, or_expr, fetch_limit)

            if not rows:
                like_term = f"%{tokens[0]}%"
                found = conn.execute(
                    """
                    SELECT * FROM greenbook_products
                    WHERE product_name LIKE ?
                       OR ingredient_name LIKE ?
                       OR applicant_name LIKE ?
                       OR nafdac_number LIKE ?
                    LIMIT ?;
                    """,
                    (like_term, like_term, like_term, like_term, fetch_limit),
                ).fetchall()
                rows = [dict(r) for r in found]

        return _prioritize_manufacturer(rows, manufacturer)[:limit]

    def count(self) -> int:
        if not self.db_path.exists():
            return 0
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM greenbook_products;").fetchone()
            return int(row["c"]) if row else 0

    def metadata(self) -> Dict[str, str]:
        if not self.db_path.exists():
            return {}
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM sync_meta;").fetchall()
        return {str(r["key"]): str(r["value"]) for r in rows}


def _fts_query(conn: sqlite3.Connection, match_expr: str, limit: int) -> List[Dict]:
    try:
        rows = conn.execute(
            """
            SELECT p.*
            FROM greenbook_fts
            JOIN greenbook_products p ON p.product_id = greenbook_fts.rowid
            WHERE greenbook_fts MATCH ?
            ORDER BY bm25(greenbook_fts) ASC
            LIMIT ?;
            """,
            (match_expr, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def _prioritize_manufacturer(rows: List[Dict], manufacturer: Optional[str]) -> List[Dict]:
    if not manufacturer or not rows:
        return rows
    tokens = [t.lower() for t in _TOKEN_RE.findall(manufacturer) if len(t) >= 3]
    if not tokens:
        return rows

    def _score(row: Dict) -> int:
        applicant = str(row.get("applicant_name") or "").lower()
        return -sum(1 for tok in tokens if tok in applicant)

    return sorted(rows, key=_score)
