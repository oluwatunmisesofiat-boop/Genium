#!/usr/bin/env python3
"""Promote a staged packaging reference image to `approved` in `product.json`.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/promote_reference.py augmentin_625_mg 0c0d49be72950182 --by "Pharm. Reviewer"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from intelligence.synthesizer import compute_reference_level

REPO_ROOT = Path(__file__).resolve().parent.parent


def promote_reference_image(
    product_json_path: Path,
    image_id: str,
    *,
    approved_by: str,
) -> Dict:
    """Move a matching reference image from `staging` to `approved` and mark its edition approved."""
    path = Path(product_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Product dossier not found: {path}")
    reviewer = (approved_by or "").strip()
    if not reviewer:
        raise ValueError("approved_by must be non-empty")

    data = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    matched_img = None

    for ver in data.get("packaging_versions") or []:
        for img in ver.get("images") or []:
            if str(img.get("image_id") or "") == image_id:
                img["status"] = "approved"
                img["approved_by"] = reviewer
                img["approved_at"] = now
                img["market_verified"] = True
                ver["status"] = "approved"
                matched_img = img
                break
        if matched_img is not None:
            break

    if matched_img is None:
        raise KeyError(f"Image id {image_id!r} not found in {path}")

    data["reference_level"] = compute_reference_level(data.get("packaging_versions") or [])
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return matched_img


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a staging reference image to approved")
    parser.add_argument("product_slug", help="Product folder slug under data/products/")
    parser.add_argument("image_id", help="16-char image_id to promote")
    parser.add_argument("--by", required=True, dest="approved_by", help="Reviewer name or handle")
    parser.add_argument(
        "--products-dir",
        type=Path,
        default=REPO_ROOT / "data" / "products",
        help="Base products directory (default: data/products)",
    )
    args = parser.parse_args()

    target = args.products_dir / args.product_slug / "product.json"
    try:
        promoted = promote_reference_image(target, args.image_id, approved_by=args.approved_by)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Approved image {promoted['image_id']} ({promoted.get('view_type')}) "
        f"in {target} by {promoted['approved_by']} at {promoted['approved_at']}"
    )


if __name__ == "__main__":
    main()
