#!/usr/bin/env python3
"""Mirror the NAFDAC Greenbook registered product database into local SQLite."""
from __future__ import annotations

import argparse
from pathlib import Path

from registry import GreenbookStore, default_db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync NAFDAC Greenbook into local SQLite")
    parser.add_argument("--db", type=Path, default=default_db_path(), help="Target SQLite DB path")
    parser.add_argument("--batch-size", type=int, default=500, help="AJAX batch size per request")
    parser.add_argument("--limit", type=int, default=None, help="Optional max records to sync")
    args = parser.parse_args()

    store = GreenbookStore(args.db)
    print(f"Syncing NAFDAC Greenbook into {store.db_path} (batch_size={args.batch_size})...")

    def _on_progress(fetched: int, total: int | None) -> None:
        total_str = str(total) if total is not None else "?"
        print(f"  synced {fetched}/{total_str} records...")

    summary = store.sync_from_remote(
        batch_size=args.batch_size,
        max_records=args.limit,
        progress_cb=_on_progress,
    )
    print(
        f"Done: {summary['records_stored']} records stored "
        f"(total remote: {summary['records_total']}) at {summary['db_path']}"
    )


if __name__ == "__main__":
    main()
