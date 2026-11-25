#!/usr/bin/env python3
"""Run a live end-to-end GENIE AI discovery + synthesis test (Steps 1 -> 2 -> 3).

Usage examples:
  PYTHONPATH=. .venv/bin/python scripts/run_discovery.py --name "Augmentin 625 mg"
  PYTHONPATH=. .venv/bin/python scripts/run_discovery.py --nafdac "A4-6238"
  PYTHONPATH=. .venv/bin/python scripts/run_discovery.py --name "Coartem 20/120" --no-images
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

from discovery import DiscoveryBundle, DiscoverySeed, discover
from intelligence import CandidateProductDossier, LocalImageStorage, synthesize_dossier
from intelligence.synthesizer import _format_greenbook_source
from registry import TrustedSourceStore

REPO_ROOT = Path(__file__).resolve().parent.parent
_WS_RE = re.compile(r"\s+")
_SAFE_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
_TYPO_TRANS = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)


def load_dotenv(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a local .env file without overriding existing env vars."""
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").translate(_TYPO_TRANS)).strip()


def _slugify(text: str) -> str:
    return _SAFE_SLUG_RE.sub("_", (text or "").strip().lower()).strip("_") or "run"


def audit_claims(bundle: DiscoveryBundle, dossier: CandidateProductDossier) -> Dict:
    """Preview Step 4's deterministic verbatim-quote check against fetched sources & Greenbook."""
    corpus_by_url: Dict[str, str] = {}
    for match in bundle.greenbook_matches:
        url = str(match.get("detail_url") or "").strip()
        if url:
            corpus_by_url[url] = _norm(_format_greenbook_source(match))

    for src in bundle.sources:
        parts = [src.content or ""]
        if src.product:
            parts.append(json.dumps(src.product, ensure_ascii=False))
        if src.identifiers:
            parts.append(json.dumps(src.identifiers, ensure_ascii=False))
        corpus_by_url[src.url] = _norm("\n".join(parts))

    matched = 0
    unmatched: List[Dict] = []
    for claim in dossier.claims:
        quote = _norm(claim.verbatim_quote)
        source_text = corpus_by_url.get(claim.source_url, "")
        if quote and source_text and quote in source_text:
            matched += 1
        else:
            unmatched.append(
                {
                    "field_name": claim.field_name,
                    "value": claim.value,
                    "source_url": claim.source_url,
                    "verbatim_quote": claim.verbatim_quote,
                    "source_found": bool(source_text),
                }
            )
    return {
        "total": len(dossier.claims),
        "matched": matched,
        "unmatched": unmatched,
    }


def _db_record_payload(dossier: CandidateProductDossier) -> Dict:
    """Build the clean, DB-ready product dossier JSON (no rejected images or debug noise)."""
    return {
        "brand_name": dossier.brand_name,
        "generic_name": dossier.generic_name,
        "active_ingredients": dossier.active_ingredients,
        "strength": dossier.strength,
        "dosage_form": dossier.dosage_form,
        "pack_size": dossier.pack_size,
        "manufacturer": dossier.manufacturer,
        "nafdac_number": dossier.nafdac_number,
        "registration_status": dossier.registration_status,
        "gtins": dossier.gtins,
        "shelf_life_months": dossier.shelf_life_months,
        "storage_statement": dossier.storage_statement,
        "packaging_description": dossier.packaging_description,
        "color_scheme": dossier.color_scheme,
        "distinctive_features": dossier.distinctive_features,
        "expected_security_features": [f.model_dump() for f in dossier.expected_security_features],
        "known_counterfeit_tells": [t.model_dump() for t in dossier.known_counterfeit_tells],
        "recall_alerts": dossier.recall_alerts,
        "version_notes": dossier.version_notes,
        "claims": [c.model_dump() for c in dossier.claims],
        "dossier_status": dossier.dossier_status,
        "reference_level": dossier.reference_level,
        "packaging_versions": [v.model_dump() for v in dossier.packaging_versions],
    }


def _print_summary(
    bundle: DiscoveryBundle,
    dossier: CandidateProductDossier,
    quote_audit: Dict,
    storage: LocalImageStorage,
    out_path: Path,
) -> None:
    print("\n" + "=" * 72)
    print("GENIE AI — LIVE DISCOVERY & SYNTHESIS REPORT")
    print("=" * 72)

    print("\n[1] Seed & Local NAFDAC Greenbook")
    print(f"  Original seed : {bundle.seed.to_dict()}")
    print(f"  Enriched seed : {bundle.enriched_seed.to_dict()}")
    print(f"  Greenbook hits: {len(bundle.greenbook_matches)}")
    for m in bundle.greenbook_matches[:3]:
        print(
            f"    - {m.get('product_name')} | NAFDAC={m.get('nafdac_number')} | "
            f"Strength={m.get('strength')} | Applicant={m.get('applicant_name')} | "
            f"SmPC={'yes' if m.get('smpc_url') else 'no'}"
        )

    print("\n[2] Multi-Angle Search & Extraction")
    print(
        f"  Queries={bundle.stats.get('queries_executed')} | "
        f"Raw hits={bundle.stats.get('raw_hits')} | "
        f"Selected={bundle.stats.get('urls_selected')} | "
        f"Used={bundle.stats.get('pages_extracted')} | "
        f"Dropped(off-topic)={bundle.stats.get('pages_dropped_off_topic', 0)} | "
        f"Failed={bundle.stats.get('pages_failed')} | "
        f"ImgSearch={bundle.stats.get('image_candidates_found', 0)} | "
        f"Elapsed={bundle.stats.get('elapsed_s')}s"
    )
    if bundle.search_errors:
        print("  Search errors:")
        for err in bundle.search_errors:
            print(f"    ! [{err.get('angle')}] {err.get('error')}")
    print("  Used sources:")
    for s in bundle.sources:
        tag = " [SmPC PDF]" if s.is_smpc_pdf else ""
        print(
            f"    - Tier {s.trust_tier} ({s.trust_label}){tag} | {s.url} "
            f"| chars={len(s.content)} | imgs={len(s.images)} | angles={s.angles}"
        )
    if bundle.dropped_sources:
        print("  Dropped off-topic sources (not saved to dossier):")
        for d in bundle.dropped_sources:
            print(f"    ~ [{d.get('reason')}] {d.get('url')} ({d.get('title')})")
    if bundle.failed_sources:
        print("  Failed sources:")
        for f in bundle.failed_sources:
            print(f"    x {f.get('url')} ({f.get('reason')}: {f.get('error')})")

    print(f"\n[3] Synthesized Product Dossier (model={dossier.model_used}, {dossier.stats.get('elapsed_s')}s)")
    print(f"  Brand Name     : {dossier.brand_name}")
    print(f"  Generic Name   : {dossier.generic_name}")
    print(f"  Ingredients    : {dossier.active_ingredients}")
    print(f"  Strength       : {dossier.strength}")
    print(f"  Dosage Form    : {dossier.dosage_form}")
    print(f"  Pack Size      : {dossier.pack_size}")
    print(f"  Manufacturer   : {dossier.manufacturer}")
    print(f"  NAFDAC Number  : {dossier.nafdac_number} (Status: {dossier.registration_status})")
    print(f"  GTINs          : {dossier.gtins}")
    print(f"  Shelf Life     : {dossier.shelf_life_months} months | Storage: {dossier.storage_statement}")
    print(f"  Packaging Desc : {dossier.packaging_description}")
    print(f"  Color Scheme   : {dossier.color_scheme}")
    print(f"  Features       : {dossier.distinctive_features}")
    print(f"  Sec. Features  : {len(dossier.expected_security_features)} extracted")
    print(f"  Fake Tells     : {len(dossier.known_counterfeit_tells)} extracted")
    print(f"  Recall Alerts  : {dossier.recall_alerts}")
    print(f"  Version Notes  : {dossier.version_notes}")
    print(f"  Ref. Level     : Level {dossier.reference_level} (approved images only)")

    gate = dossier.stats.get("evidence_gate") or {}
    print(
        f"\n[4] Deterministic Evidence Gate: status={dossier.dossier_status} | "
        f"quotes={quote_audit['matched']}/{quote_audit['total']} | "
        f"quarantined={len(gate.get('quarantined_items') or [])} | "
        f"conflicts={len(gate.get('tier1_conflicts') or [])}"
    )
    for reason in gate.get("blocking_reasons") or []:
        print(f"    * gate_note: {reason}")
    for miss in quote_audit["unmatched"][:5]:
        print(
            f"    ! UNMATCHED [{miss['field_name']}] value={miss['value']!r} "
            f"source_found={miss['source_found']} quote={miss['verbatim_quote']!r}"
        )

    print(
        f"\n[5] Packaging Versions: {len(dossier.packaging_versions)} edition(s), "
        f"{len(dossier.reference_images)} verified image(s), {len(dossier.rejected_images)} rejected"
    )
    funnel = dossier.stats.get("image_funnel") or {}
    if funnel:
        print(
            f"  Funnel: raw={funnel.get('raw_total', 0)} "
            f"(pages={funnel.get('raw_from_pages', 0)}, search={funnel.get('raw_from_search', 0)}) "
            f"-> dropped_alert={funnel.get('dropped_disqualified', 0)}, "
            f"dropped_untrusted={funnel.get('dropped_untrusted', 0)} "
            f"-> trusted={funnel.get('after_trusted', 0)} "
            f"-> downloaded={funnel.get('downloaded', 0)} "
            f"-> vision_ok={funnel.get('passed_vision', 0)} "
            f"-> saved={funnel.get('saved', 0)}"
        )
    for ver in dossier.packaging_versions:
        role = "PRIMARY" if ver.is_primary else "ALTERNATE"
        reg = " [NAFDAC/MAS markers]" if ver.has_local_regulatory_markers else ""
        summary = f" — {ver.summary}" if ver.summary else ""
        print(f"  * {ver.edition_id} [{role}]{reg}{summary}")
        for img in ver.images:
            full_path = storage.resolve(img.storage_path)
            src_tag = (
                f" | trusted={img.captured_from_domain} ({img.captured_from_type})"
                if img.captured_from_domain
                else ""
            )
            print(
                f"      + [{img.view_type}] {img.width}x{img.height} dhash={img.dhash} "
                f"| tier={img.source_trust_tier}{src_tag} | saved={full_path}"
            )
    for rej in dossier.rejected_images:
        detail = f" ({rej['detail']})" if rej.get("detail") else ""
        print(f"    - REJECTED [{rej.get('reason')}]{detail} | {rej.get('url')}")

    print(f"\nLean JSON saved to: {out_path}")
    print("=" * 72 + "\n")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run live GENIE AI discovery & synthesis")
    parser.add_argument("--name", default=None, help="Product brand/generic name")
    parser.add_argument("--nafdac", default=None, help="NAFDAC registration number (e.g. A4-6238)")
    parser.add_argument("--manufacturer", default=None, help="Optional manufacturer hint")
    parser.add_argument("--gtin", default=None, help="Optional GTIN/EAN/UPC barcode")
    parser.add_argument("--max-urls", type=int, default=6, help="Max URLs to extract (default: 6)")
    parser.add_argument("--max-results", type=int, default=6, help="Max search hits per angle")
    parser.add_argument(
        "--max-stage",
        choices=["plain", "tls", "browser", "stealth"],
        default=None,
        help="Optional scraper load-ladder cap",
    )
    parser.add_argument("--no-images", action="store_true", help="Skip image download & multimodal check")
    parser.add_argument(
        "--allow-untrusted-images",
        action="store_true",
        help="Debug flag: allow image candidates outside verified trusted_sources",
    )
    parser.add_argument("--no-smpc", action="store_true", help="Skip fetching official Greenbook SmPC PDF")
    parser.add_argument("--save-raw", action="store_true", help="Include raw extracted page texts in output JSON")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON file path")
    args = parser.parse_args()

    try:
        seed = DiscoverySeed(
            product_name=args.name,
            nafdac_number=args.nafdac,
            manufacturer=args.manufacturer,
            gtin=args.gtin,
        )
    except ValueError as exc:
        parser.error(str(exc))

    trusted_store = TrustedSourceStore()
    trusted_store.ensure_seeded()

    print("Running Step 2 (Greenbook + Multi-Angle Discovery)...")
    bundle = discover(
        seed,
        trusted_store=None if args.allow_untrusted_images else trusted_store,
        max_urls=args.max_urls,
        max_results_per_query=args.max_results,
        max_stage=args.max_stage,
        include_smpc_pdf=not args.no_smpc,
    )

    slug = _slugify(
        bundle.enriched_seed.product_name
        or bundle.enriched_seed.nafdac_number
        or args.name
        or args.nafdac
        or args.gtin
        or "product"
    )
    product_dir = (args.out.parent if args.out is not None else (REPO_ROOT / "data" / "products" / slug))
    out_path = args.out or (product_dir / "product.json")
    product_dir.mkdir(parents=True, exist_ok=True)

    print("Running Step 3 (Gemini Dossier Synthesis + Visual Verification)...", flush=True)
    storage = LocalImageStorage(product_dir)
    dossier = synthesize_dossier(
        bundle,
        storage=storage,
        product_key="images",
        verify_visuals=not args.no_images,
        trusted_store=trusted_store,
        trusted_only_images=not args.allow_untrusted_images,
        progress_cb=lambda msg: print(f"  {msg}", flush=True),
    )
    quote_audit = audit_claims(bundle, dossier)

    payload: Dict = _db_record_payload(dossier)
    if args.save_raw:
        payload["_debug_raw_bundle"] = bundle.to_dict()

    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _print_summary(bundle, dossier, quote_audit, storage, out_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
