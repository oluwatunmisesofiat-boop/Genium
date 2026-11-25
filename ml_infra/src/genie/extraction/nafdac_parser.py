"""NAFDAC Greenbook local SQLite + FTS5 mirror and client."""
from __future__ import annotations

import html
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

_NAFDAC_QUERY_RE = re.compile(r"^(?:[A-Z]\d|\d{2})\s*[-–—]\s*\d{4,6}[A-Z]?$", re.IGNORECASE)
_DASH_WS_RE = re.compile(r"\s*[-–—]\s*")


def normalize_nafdac_number(raw: object) -> str:
    text = html.unescape(str(raw or "")).strip().upper()
    text = _DASH_WS_RE.sub("-", text)
    # Normalize hyphenless NAFDAC codes like A44724 -> A4-4724
    m = re.match(r"^([A-Z]\d)(\d{4,6}[A-Z]?)$", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return text


class GreenbookRegistry:
    """Provides fast local lookups against the NAFDAC Greenbook SQLite database."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or "./data/greenbook/greenbook.db")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def lookup_number(self, nafdac_number: str) -> Optional[Dict]:
        norm = normalize_nafdac_number(nafdac_number)
        if not self.db_path.is_file():
            return None

        with self._connection() as conn:
            cur = conn.execute(
                "SELECT * FROM greenbook WHERE nafdac_number = ? LIMIT 1",
                (norm,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def search_brand(self, query: str, limit: int = 5) -> List[Dict]:
        if not self.db_path.is_file():
            return []

        with self._connection() as conn:
            cur = conn.execute(
                "SELECT * FROM greenbook WHERE product_name LIKE ? LIMIT ?",
                (f"%{query}%", limit),
            )
            return [dict(r) for r in cur.fetchall()]
