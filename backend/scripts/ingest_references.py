#!/usr/bin/env python3
"""Ingest curated reference packaging images from `data/reference_seeds.json` into `data/products/<slug>/`.

Supports three reliable, search-engine-free source modes per curated entry:
  1. `local_path`: ingest a local photo file directly (e.g. phone photos of real Nigerian stock).
  2. `image_url` (+ verified `page_url`): download a verified image URL directly.
  3. `page_url` only: open the verified pharmacy product page directly via our scraper
     and extract its main `jsonld` / `og:image` / `<img>` product photo.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/ingest_references.py --find-urls
  PYTHONPATH=. .venv/bin/python scripts/ingest_references.py --slug coartem_20_120_tablets --approve-by "Pharm. Reviewer"
  PYTHONPATH=. .venv/bin/python scripts/ingest_references.py --approve-by "Pharm. Reviewer"
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

from intelligence.images import _safe_http_download, compute_dhash, hamming_distance
from intelligence.storage import LocalImageStorage
from intelligence.synthesizer import compute_reference_level
from registry import GreenbookStore, TrustedSourceStore
from scraper.extract import extract_report
from scraper.search import search_images, search_report
from scripts.run_discovery import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS_PATH = REPO_ROOT / "data" / "reference_seeds.json"
DEFAULT_PRODUCTS_DIR = REPO_ROOT / "data" / "products"
DEFAULT_REVIEW_DIR = REPO_ROOT / "data" / "review_images"


def _bootstrap_product_json(
    product_json_path: Path,
    *,
    product_name: str,
    nafdac_number: Optional[str],
    greenbook: Optional[GreenbookStore] = None,
) -> Dict:
    """Load existing product.json or create a clean Greenbook-anchored skeleton."""
    if product_json_path.is_file():
        return json.loads(product_json_path.read_text(encoding="utf-8"))

    gb_row: Optional[Dict] = None
    if greenbook is not None:
        if nafdac_number:
            hits = greenbook.lookup_by_nafdac(nafdac_number)
            gb_row = hits[0] if hits else None
        if gb_row is None and product_name:
            hits = greenbook.search(product_name, limit=1)
            gb_row = hits[0] if hits else None

    if gb_row:
        return {
            "brand_name": gb_row.get("product_name") or product_name,
            "generic_name": gb_row.get("ingredient_name") or None,
            "active_ingredients": [
                p.strip()
                for p in str(gb_row.get("ingredient_name") or "").split(";")
                if p.strip()
            ],
            "strength": gb_row.get("strength") or None,
            "dosage_form": gb_row.get("form_name") or None,
            "pack_size": gb_row.get("pack_size") or None,
            "manufacturer": gb_row.get("applicant_name") or None,
            "nafdac_number": gb_row.get("nafdac_number") or nafdac_number,
            "registration_status": gb_row.get("status") or None,
            "packaging_description": gb_row.get("product_description") or None,
            "color_scheme": [],
            "distinctive_features": [],
            "recall_alerts": [],
            "version_notes": [],
            "claims": [],
            "packaging_versions": [],
        }

    return {
        "brand_name": product_name,
        "generic_name": None,
        "active_ingredients": [],
        "strength": None,
        "dosage_form": None,
        "pack_size": None,
        "manufacturer": None,
        "nafdac_number": nafdac_number,
        "registration_status": None,
        "packaging_description": None,
        "color_scheme": [],
        "distinctive_features": [],
        "recall_alerts": [],
        "version_notes": [],
        "claims": [],
        "packaging_versions": [],
    }


def _inspect_image_bytes(raw_bytes: bytes, header_ct: str = "") -> Tuple[int, int, str, str, str]:
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.load()
            width, height = img.size
            pil_fmt = img.format
            dhash = compute_dhash(img)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid image bytes: {exc}") from exc

    mime = (
        Image.MIME.get(pil_fmt or "")
        or (header_ct.split(";", 1)[0].strip().lower() if header_ct else "")
        or "image/jpeg"
    )
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    return width, height, mime, sha256, dhash


def _extract_top_image_from_page(
    page_url: str,
    *,
    extract_fn: Callable = extract_report,
) -> str:
    report = extract_fn([page_url], fmt="text", structured=True)
    results = list(report.get("results") or [])
    if not results:
        failed = list(report.get("failed") or [])
        reason = failed[0].get("error") if failed else "page_extraction_empty"
        raise RuntimeError(f"Failed to open {page_url}: {reason}")
    images = list(results[0].get("images") or [])
    if not images:
        raise RuntimeError(f"No product images found on {page_url}")
    return str(images[0].get("url") or "").strip()


def ingest_product_references(
    entry: Dict,
    *,
    products_dir: Path = DEFAULT_PRODUCTS_DIR,
    trusted_store: Optional[TrustedSourceStore] = None,
    greenbook: Optional[GreenbookStore] = None,
    allow_untrusted: bool = False,
    approve_by: Optional[str] = None,
    extract_fn: Callable = extract_report,
    downloader: Optional[Callable[[str], Tuple[bytes, str]]] = None,
) -> Dict:
    """Ingest all curated reference images for a single product seed entry."""
    slug = str(entry.get("slug") or "").strip()
    if not slug:
        raise ValueError("Seed entry missing 'slug'")
    product_name = str(entry.get("product_name") or slug).strip()
    nafdac_number = entry.get("nafdac_number")
    sources = list(entry.get("sources") or [])

    product_dir = Path(products_dir) / slug
    product_dir.mkdir(parents=True, exist_ok=True)
    product_json_path = product_dir / "product.json"
    storage = LocalImageStorage(product_dir)

    data = _bootstrap_product_json(
        product_json_path,
        product_name=product_name,
        nafdac_number=nafdac_number,
        greenbook=greenbook,
    )
    versions: List[Dict] = list(data.get("packaging_versions") or [])
    existing_shas = {
        str(img.get("sha256") or "")
        for v in versions
        for img in (v.get("images") or [])
    }

    added = 0
    skipped = 0
    errors: List[str] = []
    now = datetime.now(timezone.utc).isoformat()
    reviewer = (approve_by or "").strip() or None

    for src in sources:
        local_path_str = str(src.get("local_path") or "").strip()
        page_url = str(src.get("page_url") or "").strip()
        image_url = str(src.get("image_url") or "").strip()
        view_type = str(src.get("view_type") or "front_box").strip()
        edition_id = str(src.get("edition_id") or "edition_1").strip()
        pack_variant = src.get("pack_size_variant")
        summary = src.get("summary") or src.get("notes")
        has_local_markers = bool(src.get("has_local_regulatory_markers", False))

        captured_domain: Optional[str] = None
        captured_type: Optional[str] = None
        trust_tier = 2

        try:
            if local_path_str:
                lp = Path(local_path_str)
                if not lp.is_absolute():
                    lp = REPO_ROOT / lp
                if not lp.is_file():
                    raise FileNotFoundError(f"Local image not found: {lp}")
                raw_bytes = lp.read_bytes()
                header_ct = ""
                source_url = f"file://{lp.resolve()}"
                source_page_url = page_url or source_url
                captured_domain = "direct_capture"
                captured_type = "direct_capture"
                trust_tier = 1
            else:
                target_for_trust = page_url or image_url
                if not target_for_trust:
                    raise ValueError("Source must specify local_path, page_url, or image_url")
                if trusted_store is not None:
                    match = trusted_store.is_verified_image_source(target_for_trust)
                    if match is None and not allow_untrusted:
                        raise PermissionError(
                            f"Source {target_for_trust!r} is not a verified trusted_source"
                        )
                    if match is not None:
                        captured_domain = match["domain"]
                        captured_type = match["source_type"]
                        trust_tier = 1 if match["source_type"] == "manufacturer" else 2

                if not image_url:
                    image_url = _extract_top_image_from_page(page_url, extract_fn=extract_fn)

                if downloader is not None:
                    raw_bytes, header_ct = downloader(image_url)
                else:
                    raw_bytes, header_ct = _safe_http_download(
                        image_url, timeout=12.0, max_bytes=8_000_000
                    )
                source_url = image_url
                source_page_url = page_url or image_url

            width, height, mime, sha256, dhash = _inspect_image_bytes(raw_bytes, header_ct)
            if sha256 in existing_shas:
                skipped += 1
                continue

            rel_path = storage.save("images", sha256, raw_bytes, mime)
            img_record = {
                "image_id": sha256[:16],
                "storage_path": rel_path,
                "source_url": source_url,
                "source_page_url": source_page_url,
                "source_trust_tier": trust_tier,
                "sha256": sha256,
                "dhash": dhash,
                "width": width,
                "height": height,
                "mime_type": mime,
                "view_type": view_type,
                "packaging_edition_id": edition_id,
                "pack_size_variant": pack_variant,
                "is_real_photo": True,
                "has_local_regulatory_markers": has_local_markers,
                "visible_cues": list(src.get("visible_cues") or []),
                "visible_text_snippets": list(src.get("visible_text_snippets") or []),
                "captured_from_domain": captured_domain,
                "captured_from_type": captured_type,
                "captured_at": now,
                "market_verified": bool(has_local_markers or reviewer),
                "approved_by": reviewer,
                "approved_at": now if reviewer else None,
                "status": "approved" if reviewer else "staging",
            }

            bucket = next(
                (
                    v
                    for v in versions
                    if v.get("edition_id") == edition_id
                    and (v.get("pack_size_variant") or "") == (pack_variant or "")
                ),
                None,
            )
            if bucket is None:
                bucket = {
                    "edition_id": edition_id,
                    "pack_size_variant": pack_variant,
                    "summary": summary,
                    "is_primary": len(versions) == 0,
                    "has_local_regulatory_markers": has_local_markers,
                    "status": "approved" if reviewer else "staging",
                    "images": [],
                }
                versions.append(bucket)
            elif reviewer:
                bucket["status"] = "approved"
            if has_local_markers:
                bucket["has_local_regulatory_markers"] = True

            bucket["images"].append(img_record)
            existing_shas.add(sha256)
            added += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{page_url or image_url or local_path_str}: {exc}")

    data["packaging_versions"] = versions
    data["reference_level"] = compute_reference_level(versions)
    product_json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "slug": slug,
        "product_json": str(product_json_path),
        "added": added,
        "skipped_duplicate": skipped,
        "errors": errors,
    }


def _find_candidate_pharmacy_urls(
    seeds: Dict,
    *,
    trusted_store: TrustedSourceStore,
    slug_filter: Optional[str] = None,
) -> None:
    """One-time helper: search verified Nigerian pharmacies for product pages to paste into reference_seeds.json."""
    domains = trusted_store.get_verified_image_domains(country="NG")[:10]
    top_domains = [
        d
        for d in ("healthplusnigeria.com", "drugstore.ng", "hubpharmafrica.com", "purelifehealth.io", "sanlivepharmacy.com")
        if d in domains
    ] or domains[:5]

    for entry in seeds.get("products") or []:
        slug = entry.get("slug")
        if slug_filter and slug != slug_filter:
            continue
        name = entry.get("product_name") or slug
        print(f"\n=== {name} ({slug}) ===")
        try:
            res = search_report(name, include_domains=top_domains, max_results=5, country="ng")
            hits = res.get("results") or []
            if not hits:
                print("  (no hits on top 5 verified pharmacies)")
            for h in hits:
                print(f"  - {h.get('title')} -> {h.get('url')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! search error: {exc}")


_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def _safe_host_label(url: str) -> str:
    from urllib.parse import urlparse

    host = (urlparse(url or "").hostname or "web").lower().removeprefix("www.")
    return "".join(ch if ch.isalnum() else "_" for ch in host).strip("_")[:32] or "web"


def fetch_review_images_for_product(
    entry: Dict,
    *,
    review_dir: Path = DEFAULT_REVIEW_DIR,
    trusted_store: Optional[TrustedSourceStore] = None,
    max_images: int = 8,
    search_images_fn: Callable = search_images,
    downloader: Optional[Callable[[str], Tuple[bytes, str]]] = None,
) -> Dict:
    """Download candidate packaging images into `data/review_images/<slug>/` for manual inspection.

    Does NOT write to `product.json`. The user inspects the folder, deletes any bad images,
    and then runs `--ingest-review` to import only the surviving legit photos.
    """
    from discovery.trust import classify_url, is_disqualified_image_source

    slug = str(entry.get("slug") or "").strip()
    name = str(entry.get("product_name") or slug).strip()
    out_dir = Path(review_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pull candidates from Serper/DDGS image search (Nigeria-scoped + clean base name)
    raw_hits: List[Dict] = []
    for query in (f"{name} Nigeria pharmacy", f"{name} pack box"):
        try:
            hits = list(search_images_fn(query, max_results=10, country="ng") or [])
            raw_hits.extend(hits)
        except Exception:  # noqa: BLE001
            continue

    manifest_entries: List[Dict] = []
    seen_urls: set[str] = set()
    seen_shas: set[str] = set()
    seen_dhashes: List[str] = []
    saved_count = 0

    for hit in raw_hits:
        if saved_count >= max_images:
            break
        img_url = str(hit.get("url") or "").strip()
        page_url = str(hit.get("source_page_url") or "").strip()
        title = str(hit.get("title") or "").strip()
        if not img_url or img_url in seen_urls:
            continue
        seen_urls.add(img_url)

        tier, _label = classify_url(page_url or img_url)
        if tier == 0:
            continue
        if is_disqualified_image_source(img_url, page_url, title):
            continue

        try:
            if downloader is not None:
                raw_bytes, header_ct = downloader(img_url)
            else:
                raw_bytes, header_ct = _safe_http_download(img_url, timeout=8.0, max_bytes=6_000_000)
            width, height, mime, sha256, dhash = _inspect_image_bytes(raw_bytes, header_ct)
        except Exception:  # noqa: BLE001
            continue

        if width < 300 or height < 300:
            continue
        if max(width / height, height / width) > 3.5:
            continue
        if sha256 in seen_shas:
            continue
        if any(hamming_distance(dhash, prev) <= 6 for prev in seen_dhashes):
            continue

        trusted_match = (
            trusted_store.is_verified_image_source(page_url or img_url)
            if trusted_store is not None
            else None
        )
        seen_shas.add(sha256)
        seen_dhashes.append(dhash)
        saved_count += 1

        ext = _MIME_TO_EXT.get(mime, ".jpg")
        host_tag = _safe_host_label(page_url or img_url)
        trust_prefix = "TRUSTED_" if trusted_match else ""
        filename = f"{saved_count:02d}_{trust_prefix}{host_tag}_{sha256[:8]}{ext}"
        (out_dir / filename).write_bytes(raw_bytes)
        manifest_entries.append(
            {
                "filename": filename,
                "source_url": img_url,
                "source_page_url": page_url,
                "title": title,
                "width": width,
                "height": height,
                "sha256": sha256,
                "dhash": dhash,
                "captured_from_domain": trusted_match["domain"] if trusted_match else None,
                "captured_from_type": trusted_match["source_type"] if trusted_match else None,
                "is_trusted_source": bool(trusted_match),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "slug": slug,
                "product_name": name,
                "nafdac_number": entry.get("nafdac_number"),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "images": manifest_entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"slug": slug, "review_dir": str(out_dir), "saved": saved_count}


def ingest_from_review_folder(
    entry: Dict,
    *,
    review_dir: Path = DEFAULT_REVIEW_DIR,
    products_dir: Path = DEFAULT_PRODUCTS_DIR,
    greenbook: Optional[GreenbookStore] = None,
    approve_by: Optional[str] = None,
) -> Dict:
    """Ingest ONLY the images the user kept inside `data/review_images/<slug>/` after manual inspection."""
    slug = str(entry.get("slug") or "").strip()
    folder = Path(review_dir) / slug
    if not folder.is_dir():
        return {"slug": slug, "added": 0, "skipped_duplicate": 0, "errors": [f"Review folder not found: {folder}"]}

    manifest_by_file: Dict[str, Dict] = {}
    manifest_path = folder / "manifest.json"
    if manifest_path.is_file():
        mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in mdata.get("images") or []:
            manifest_by_file[str(item.get("filename") or "")] = item

    image_files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    sources: List[Dict] = []
    for p in image_files:
        meta = manifest_by_file.get(p.name) or {}
        sources.append(
            {
                "local_path": str(p.resolve()),
                "page_url": meta.get("source_page_url") or "",
                "view_type": "front_box",
                "edition_id": "edition_1",
                "notes": f"Manually verified from review folder ({p.name})",
            }
        )

    temp_entry = dict(entry)
    temp_entry["sources"] = sources
    return ingest_product_references(
        temp_entry,
        products_dir=products_dir,
        greenbook=greenbook,
        allow_untrusted=True,
        approve_by=approve_by,
    )


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Download for manual review and ingest curated packaging references")
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS_PATH, help="Path to reference_seeds.json")
    parser.add_argument("--products-dir", type=Path, default=DEFAULT_PRODUCTS_DIR, help="Output products dir")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR, help="Local manual review dir")
    parser.add_argument("--slug", default=None, help="Only process a single product slug")
    parser.add_argument("--max-images", type=int, default=8, help="Max candidate images per drug for --fetch-review")
    parser.add_argument("--approve-by", default=None, help="Auto-mark ingested references as approved by reviewer")
    parser.add_argument("--allow-untrusted", action="store_true", help="Allow URLs outside verified trusted_sources")
    parser.add_argument(
        "--fetch-review",
        action="store_true",
        help="Download candidate images into data/review_images/<drug_slug>/ for manual inspection (does NOT touch product.json)",
    )
    parser.add_argument(
        "--ingest-review",
        action="store_true",
        help="Ingest only the surviving images kept in data/review_images/<drug_slug>/ after manual inspection",
    )
    parser.add_argument(
        "--find-urls",
        action="store_true",
        help="Search verified Nigerian pharmacies and print candidate product URLs to copy into reference_seeds.json",
    )
    args = parser.parse_args()

    if not args.seeds.is_file():
        print(f"Seeds file not found: {args.seeds}", file=sys.stderr)
        sys.exit(1)

    seeds = json.loads(args.seeds.read_text(encoding="utf-8"))
    trusted_store = TrustedSourceStore()
    trusted_store.ensure_seeded()

    if args.find_urls:
        _find_candidate_pharmacy_urls(seeds, trusted_store=trusted_store, slug_filter=args.slug)
        return

    if args.fetch_review:
        for entry in seeds.get("products") or []:
            if args.slug and entry.get("slug") != args.slug:
                continue
            res = fetch_review_images_for_product(
                entry,
                review_dir=args.review_dir,
                trusted_store=trusted_store,
                max_images=args.max_images,
            )
            print(f"[{res['slug']}] downloaded {res['saved']} image(s) for review -> {res['review_dir']}")
        print("\nInspect the folders above, delete any bad/wrong images, then run --ingest-review --approve-by <your_name>.")
        return

    gb_path = REPO_ROOT / "data" / "greenbook.db"
    greenbook = GreenbookStore(gb_path) if gb_path.is_file() else None

    if args.ingest_review:
        for entry in seeds.get("products") or []:
            if args.slug and entry.get("slug") != args.slug:
                continue
            res = ingest_from_review_folder(
                entry,
                review_dir=args.review_dir,
                products_dir=args.products_dir,
                greenbook=greenbook,
                approve_by=args.approve_by,
            )
            print(
                f"[{res['slug']}] ingested={res['added']} skipped={res['skipped_duplicate']} "
                f"errors={len(res['errors'])} -> {res.get('product_json', '')}"
            )
            for err in res["errors"]:
                print(f"  ! {err}")
        return

    for entry in seeds.get("products") or []:
        if args.slug and entry.get("slug") != args.slug:
            continue
        res = ingest_product_references(
            entry,
            products_dir=args.products_dir,
            trusted_store=trusted_store,
            greenbook=greenbook,
            allow_untrusted=args.allow_untrusted,
            approve_by=args.approve_by,
        )
        print(
            f"[{res['slug']}] added={res['added']} skipped={res['skipped_duplicate']} "
            f"errors={len(res['errors'])} -> {res['product_json']}"
        )
        for err in res["errors"]:
            print(f"  ! {err}")


if __name__ == "__main__":
    main()
