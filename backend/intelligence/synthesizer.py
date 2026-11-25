"""Structured dossier synthesis and multimodal packaging image verification."""
from __future__ import annotations

import json
import re
import time
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from discovery.models import DiscoveryBundle
from discovery.trust import is_disqualified_image_source

from .images import (
    DownloadedImage,
    collect_candidate_image_urls,
    download_and_deduplicate_images,
    prepare_image_for_llm,
)
from .llm import LLMProvider, get_default_provider
from .schemas import (
    BatchImageVerificationResponse,
    CandidateProductDossier,
    CounterfeitTell,
    ExtractedDossierResponse,
    PackagingVersion,
    SecurityFeatureSpec,
    VerifiedReferenceImage,
)
from .storage import ImageStorage, LocalImageStorage

DOSSIER_SYSTEM_INSTRUCTION = """You are GENIE AI's pharmaceutical intelligence extraction engine.
Your job is to synthesize a structured Product Dossier ONLY from the provided NAFDAC Greenbook records and scraped web/SmPC sources.

Strict grounding rules:
1. Never invent, extrapolate, or guess drug details, NAFDAC registration numbers, GTINs, shelf life, strengths, pack sizes, or manufacturers. If a field is not supported by the sources, leave it null or [].
2. For EVERY populated field (such as brand_name, generic_name, active_ingredients, strength, dosage_form, pack_size, manufacturer, nafdac_number, registration_status, shelf_life_months, storage_statement, packaging_description, recall_alerts, version_notes), emit at least one entry in `claims`.
3. Every `EvidenceClaim` MUST specify:
   - `field_name`: the schema field name it supports
   - `value`: the concise extracted value as a string
   - `source_url`: the exact `source_url` from the prompt where this evidence came from
   - `verbatim_quote`: an EXACT, contiguous substring copied character-for-character from that source's text (do not paraphrase or alter numbers).
4. Extract SmPC technical specs whenever present in `smpc_extracted_sections` or source content:
   - `shelf_life_months`: integer months (e.g. "2 years" -> 24, "36 months" -> 36).
   - `storage_statement`: exact storage condition (e.g. "Do not store above 30°C. Store in the original package.").
5. Extract non-NAFDAC intelligence when supported by sources:
   - `expected_security_features`: physical/anti-counterfeit cues (tablet debossing like 'NC'/'CG', scratch PIN, hologram, tamper seal, QR code, braille). Set `verifiable=true` ONLY for features checkable against an external authority (e.g. scratch PIN or QR resolving to manufacturer domain).
   - `known_counterfeit_tells`: actively mine `regulatory_alerts` and `packaging_alerts` sources for what seized/falsified packs of this drug got wrong — compromised batch numbers (`affected_batches`), fake MFG/EXP dates, spelling errors, wrong foil/box colors, or missing security features. Tag `source_kind="nafdac_derived"` if the source is on `nafdac.gov.ng`, otherwise `"independent"`.
6. Prefer Tier 1 regulatory/Greenbook and official manufacturer sources over Tier 2/3 when summarizing canonical identity, and surface any recall alerts or packaging change notices in `recall_alerts` and `version_notes`."""

IMAGE_VERIFY_SYSTEM_INSTRUCTION = """You are GENIE AI's pharmaceutical visual verification specialist.
Evaluate each candidate image strictly against the target pharmaceutical product AND compare candidate images against each other to separate distinct packaging editions.

Rules for every `candidate_index`:
1. Set `is_product_packaging=true` ONLY if the image clearly depicts actual consumer retail pharmaceutical packaging (box, bottle, blister pack, vial, sachet, or package leaflet). Reject website logos, icons, generic pill stock photos, doctor/lifestyle photos, marketing banners, seized/evidence piles, damaged/ripped cartons on a table or floor, and hospital/institutional multi-strip bulk dispenser boxes (e.g. "30 blister strips") when the target is a consumer retail pack.
2. Check exact formulation and pack size against the target product:
   - Set `matches_target_product=false` if the box shows a different formulation (e.g. "Dispersible", "Syrup", "Suspension", "Duo", "Forte", "Baby/Paediatric" when target is regular Tablet, or vice versa) OR a pack count outside the target pack_size.
   - Always populate `detected_formulation` (e.g. "tablet", "dispersible tablet", "syrup", "caplet"), `detected_pack_count` (integer total tablets/units printed on the pack, e.g. 14, 18, 24, 6, 12), and `pack_size_variant` (e.g. "18 tablets", "24 tablets").
3. Set `is_real_photo=true` for real photographs of physical packaging — INCLUDING clean studio or pharmacy e-commerce packshots on a plain white background. Set `is_real_photo=false` ONLY for obvious 3D CGI renders, vector mockups, or flat graphic illustrations.
4. Set `has_watermark=true` if ANY watermark, app badge (`Photoroom`, `Shutterstock`, etc.), or overlaid pharmacy logo/text stamp appears anywhere on the image; otherwise `false`.
5. Set `has_local_regulatory_markers=true` if the image visibly shows Nigerian regulatory/authentication markings such as a `NAFDAC` registration number, `Scratch PIN` / Sproxil / MAS authentication sticker, or Nigerian distributor/importer markings.
6. Group visually matching packaging cartons using `packaging_edition_id` (`edition_1`, `edition_2`, etc.):
   - Assign the SAME `packaging_edition_id` when two images show the front and back/side of the EXACT SAME carton design (same pack size, color scheme, logo style, and security label style).
   - Assign a DIFFERENT `packaging_edition_id` when pack size, carton color scheme, logo style/placement, or security labels differ, and describe the edition concisely in `edition_summary`.
7. Classify `view_type` accurately (`front_box`, `back_box`, `blister_pack`, `bottle`, `leaflet`, or `other`).
8. Extract visible `visible_cues` and `visible_text_snippets` (printed drug name, strength, pack count, NAFDAC number, MAN/EXP dates if readable).
9. Set `quality_ok=false` for blurry, unreadable, or cropped-out images, and provide a concise `rejection_reason` whenever any check fails."""


def _format_greenbook_source(match: Dict) -> str:
    source_url = str(match.get("detail_url") or "")
    lines = [
        f"### NAFDAC Greenbook Record (Tier 1 Regulatory Mirror)",
        f"source_url: {source_url}",
        f"Product Name: {match.get('product_name', '')}",
        f"NAFDAC Number: {match.get('nafdac_number', '')}",
        f"Active Ingredients: {match.get('ingredient_name', '')}",
        f"Strength: {match.get('strength', '')}",
        f"Dosage Form: {match.get('form_name', '')}",
        f"Route: {match.get('route_name', '')}",
        f"Applicant / Manufacturer: {match.get('applicant_name', '')}",
        f"Pack Size: {match.get('pack_size', '')}",
        f"Registration Status: {match.get('status', '')}",
        f"Approval Date: {match.get('approval_date', '')}",
        f"Expiry Date: {match.get('expiry_date', '')}",
        f"Composition: {match.get('composition', '')}",
        f"Product Description: {match.get('product_description', '')}",
    ]
    return "\n".join(line for line in lines if not line.endswith(": "))


_SMPC_SECTION_PATTERNS = (
    ("3. Pharmaceutical form", re.compile(r"(?im)(?:^|\n)\s*3\.?\s+pharmaceutical\s+form\b")),
    ("6.3 Shelf life", re.compile(r"(?im)(?:^|\n)\s*6\.3\.?\s+shelf[\s\-]*life\b|\bshelf[\s\-]*life\s*:")),
    (
        "6.4 Special precautions for storage",
        re.compile(
            r"(?im)(?:^|\n)\s*6\.4\.?\s+special\s+precautions\s+for\s+storage\b|\bstorage\s+conditions?\s*:"
        ),
    ),
    (
        "6.5 Nature and contents of container",
        re.compile(r"(?im)(?:^|\n)\s*6\.5\.?\s+nature\s+and\s+contents\s+of\s+container\b"),
    ),
)
_NEXT_SMPC_HEADING_RE = re.compile(r"(?m)\n\s*(?:\d+\.\d+|\d+\.)\s+[A-Z]")
_SHELF_LIFE_YEARS_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", re.IGNORECASE)
_SHELF_LIFE_MONTHS_RE = re.compile(r"\b(\d+)\s*(?:months?|mos?)\b", re.IGNORECASE)


def extract_smpc_key_sections(text: str, *, max_section_chars: int = 700) -> str:
    """Deterministically pull SmPC §3 / §6.3 / §6.4 / §6.5 blocks before body truncation."""
    if not text or not text.strip():
        return ""
    blocks: List[str] = []
    seen_spans: set[tuple[int, int]] = set()
    for label, pattern in _SMPC_SECTION_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        start = m.start()
        tail = text[m.end() : m.end() + max_section_chars]
        next_m = _NEXT_SMPC_HEADING_RE.search(tail)
        end = m.end() + (next_m.start() if next_m else len(tail))
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        snippet = text[start:end].strip()
        if snippet:
            blocks.append(f"[{label}]\n{snippet}")
    return "\n\n".join(blocks)


def parse_shelf_life_months(text: str) -> Optional[int]:
    """Parse an integer shelf life in months from SmPC §6.3 text (e.g. '3 years' -> 36, '24 months' -> 24)."""
    if not text:
        return None
    m_mo = _SHELF_LIFE_MONTHS_RE.search(text)
    if m_mo:
        val = int(m_mo.group(1))
        if 1 <= val <= 120:
            return val
    m_yr = _SHELF_LIFE_YEARS_RE.search(text)
    if m_yr:
        val = int(round(float(m_yr.group(1)) * 12))
        if 1 <= val <= 120:
            return val
    return None


def collect_validated_gtins(bundle: DiscoveryBundle) -> List[str]:
    """Deterministically gather GS1-validated GTINs from enriched seed and scraped sources."""
    gtins: List[str] = []

    def _add(raw: object) -> None:
        val = re.sub(r"\D", "", str(raw or ""))
        if len(val) in (8, 12, 13, 14) and val not in gtins:
            gtins.append(val)

    if bundle.enriched_seed.gtin:
        _add(bundle.enriched_seed.gtin)
    for src in bundle.sources:
        for g in (src.identifiers or {}).get("gtins") or []:
            _add(g)
        if isinstance(src.product, dict) and src.product.get("gtin"):
            _add(src.product.get("gtin"))
    return gtins


def compute_reference_level(packaging_versions: List[PackagingVersion] | List[Dict]) -> str:
    """Compute reference level A/B/C strictly from `approved` packaging images (staging never counts).

    - A: >= 2 distinct approved views (e.g. front_box + back_box/blister_pack)
    - B: exactly 1 distinct approved view (usually front_box)
    - C: no approved packaging references (degraded mode)
    """
    approved_views: set[str] = set()
    for ver in packaging_versions or []:
        imgs = ver.get("images", []) if isinstance(ver, dict) else (ver.images or [])
        for img in imgs:
            status = img.get("status") if isinstance(img, dict) else img.status
            view = img.get("view_type") if isinstance(img, dict) else img.view_type
            if status == "approved" and view and view != "other":
                approved_views.add(str(view).strip().lower())
    if len(approved_views) >= 2:
        return "A"
    if len(approved_views) == 1:
        return "B"
    return "C"


def _normalize_counterfeit_tells(tells: List[CounterfeitTell]) -> List[CounterfeitTell]:
    """Enforce honest source_kind tagging: any NAFDAC-hosted alert is `nafdac_derived`."""
    out: List[CounterfeitTell] = []
    for tell in tells:
        host = (urlparse(tell.source_url or "").hostname or "").lower()
        if host == "nafdac.gov.ng" or host.endswith(".nafdac.gov.ng"):
            out.append(tell.model_copy(update={"source_kind": "nafdac_derived"}))
        else:
            out.append(tell)
    return out


def build_dossier_prompt(
    bundle: DiscoveryBundle,
    *,
    max_chars_per_source: int = 4000,
) -> str:
    """Serialize Greenbook records and scraped discovery sources for grounded LLM synthesis."""
    sections: List[str] = [
        "## Discovery Seed",
        f"Original Seed: {json.dumps(bundle.seed.to_dict(), ensure_ascii=False)}",
        f"Enriched Seed: {json.dumps(bundle.enriched_seed.to_dict(), ensure_ascii=False)}",
    ]

    if bundle.greenbook_matches:
        sections.append("## Local Regulatory Baseline (NAFDAC Greenbook)")
        for match in bundle.greenbook_matches[:3]:
            sections.append(_format_greenbook_source(match))

    if bundle.sources:
        sections.append("## Scraped Web & SmPC Sources")
        for idx, src in enumerate(bundle.sources):
            parts = [
                f"### Source {idx + 1} (Tier {src.trust_tier} - {src.trust_label})",
                f"source_url: {src.url}",
                f"title: {src.title}",
                f"angles: {', '.join(src.angles)}",
                f"is_smpc_pdf: {src.is_smpc_pdf}",
            ]
            if src.product:
                parts.append(f"json_ld_product: {json.dumps(src.product, ensure_ascii=False)}")
            if src.identifiers and (src.identifiers.get("nafdac_numbers") or src.identifiers.get("gtins")):
                parts.append(f"extracted_identifiers: {json.dumps(src.identifiers, ensure_ascii=False)}")
            if src.is_smpc_pdf and src.content:
                smpc_blocks = extract_smpc_key_sections(src.content)
                if smpc_blocks:
                    parts.append(f"smpc_extracted_sections:\n{smpc_blocks}")
            text = (src.content or "")[:max_chars_per_source].strip()
            parts.append(f"content:\n{text}")
            sections.append("\n".join(parts))

    return "\n\n".join(sections)


def build_image_verify_prompt(
    bundle: DiscoveryBundle,
    images: List[DownloadedImage],
) -> str:
    """Build the multimodal verification prompt describing the target drug and indexed images."""
    seed = bundle.enriched_seed
    lines = [
        "## Target Pharmaceutical Product",
        f"Product Name: {seed.product_name or 'Unknown'}",
        f"Manufacturer / Applicant: {seed.manufacturer or 'Unknown'}",
        f"NAFDAC Number: {seed.nafdac_number or 'Unknown'}",
        f"GTIN: {seed.gtin or 'Unknown'}",
    ]
    if bundle.greenbook_matches:
        top = bundle.greenbook_matches[0]
        lines.extend(
            [
                f"Active Ingredients: {top.get('ingredient_name') or 'Unknown'}",
                f"Strength: {top.get('strength') or 'Unknown'}",
                f"Dosage Form: {top.get('form_name') or 'Unknown'}",
                f"Pack Size: {top.get('pack_size') or 'Unknown'}",
                f"Official Description: {top.get('product_description') or 'N/A'}",
            ]
        )

    lines.append("")
    lines.append(
        "## Candidate Images (attached in exact 0-based order below; evaluate every index)"
    )
    for idx, img in enumerate(images):
        lines.append(
            f"Candidate {idx}: url={img.url} | source_page={img.source_page_url} | "
            f"trust_tier={img.source_trust_tier} | extraction={img.extraction_source} | "
            f"dimensions={img.width}x{img.height} | alt={img.alt!r}"
        )
    return "\n".join(lines)


_DISQUALIFYING_FORMULATIONS = (
    "dispersible",
    "syrup",
    "suspension",
    "injection",
    "injectable",
    "cream",
    "ointment",
    "drops",
    "suppository",
    "chewable",
    "baby",
    "paediatric",
    "pediatric",
    "infant",
    "duo",
    "forte",
    "extra",
    "plus",
)
_STRICT_URL_FORMULATIONS = (
    "dispersible",
    "syrup",
    "suspension",
    "injection",
    "baby",
    "paediatric",
    "pediatric",
    "duo",
    "forte",
)
_WATERMARK_MARKERS = (
    "photoroom",
    "shutterstock",
    "gettyimages",
    "istock",
    "alamy",
    "dreamstime",
    "123rf",
    "depositphotos",
)
_MULTIPLIER_RE = re.compile(r"\b(\d+)\s*[x×]\s*(\d+)\b", re.IGNORECASE)
_STANDALONE_COUNT_RE = re.compile(
    r"\b(\d+)\s*(?:'s|(?:dispersible\s+|film[- ]coated\s+)?(?:tablets?|tabs?|capsules?|caps?|caplets?|sachets?|vials?|ampoules?))\b",
    re.IGNORECASE,
)
_BULK_DISPENSER_RE = re.compile(
    r"\b(?:10|15|20|24|25|30|50|100)\s*(?:blister\s+)?strips?\b|\b(?:dispenser|hospital)\s+(?:pack|carton|box)\b",
    re.IGNORECASE,
)


def parse_allowed_pack_counts(pack_size: Optional[str]) -> set[int]:
    """Parse allowed unit counts from Greenbook pack_size strings like \"3 x 6's; 4 x 6's\"."""
    if not pack_size:
        return set()
    counts: set[int] = set()
    text = str(pack_size)
    cleaned = text
    for m in _MULTIPLIER_RE.finditer(text):
        counts.add(int(m.group(1)) * int(m.group(2)))
    cleaned = _MULTIPLIER_RE.sub(" ", cleaned)
    for m in _STANDALONE_COUNT_RE.finditer(cleaned):
        counts.add(int(m.group(1)))
    return counts


def extract_pack_count_from_snippets(snippets: List[str]) -> Optional[int]:
    """Extract an explicit total pack count from OCR/visible packaging text snippets."""
    joined = " | ".join(snippets or [])
    if not joined.strip():
        return None
    for m in _MULTIPLIER_RE.finditer(joined):
        return int(m.group(1)) * int(m.group(2))
    for m in _STANDALONE_COUNT_RE.finditer(joined):
        return int(m.group(1))
    return None


def _target_context_text(bundle: DiscoveryBundle) -> tuple[str, Optional[str]]:
    parts = [bundle.enriched_seed.product_name or "", bundle.seed.product_name or ""]
    pack_size: Optional[str] = None
    if bundle.greenbook_matches:
        top = bundle.greenbook_matches[0]
        parts.extend(
            [
                str(top.get("product_name") or ""),
                str(top.get("form_name") or ""),
                str(top.get("product_description") or ""),
            ]
        )
        pack_size = str(top.get("pack_size") or "").strip() or None
    return " ".join(parts).lower(), pack_size


def _deterministic_image_rejection(
    check,
    img: DownloadedImage,
    bundle: DiscoveryBundle,
) -> Optional[Tuple[str, str]]:
    """Apply deterministic Greenbook-anchored rules that override soft LLM booleans."""
    disq_reason = is_disqualified_image_source(img.url, img.source_page_url, img.alt)
    if disq_reason:
        return disq_reason, f"Disqualified image source ({disq_reason})"

    if getattr(check, "view_type", None) == "other":
        return (
            "unusable_view_type",
            "Image view_type is 'other' rather than a clean box, blister pack, bottle, or leaflet view",
        )

    snippets_text = " ".join(check.visible_text_snippets or []).lower()
    cues_text = " ".join(check.visible_cues or []).lower()

    if check.has_watermark or any(wm in snippets_text or wm in cues_text for wm in _WATERMARK_MARKERS):
        return "watermarked", check.rejection_reason or "Watermark detected on packaging image"

    target_text, gb_pack_size = _target_context_text(bundle)
    detected_text = " ".join(
        [
            check.detected_formulation or "",
            snippets_text,
            check.pack_size_variant or "",
            check.edition_summary or "",
        ]
    ).lower()
    page_path = urlparse(img.source_page_url or "").path.lower()

    if (
        "blister strip" not in target_text
        and "dispenser" not in target_text
        and "hospital pack" not in target_text
        and _BULK_DISPENSER_RE.search(detected_text)
    ):
        return (
            "bulk_dispenser_pack",
            "Image depicts a bulk institutional dispenser carton rather than a consumer retail pack",
        )

    for term in _DISQUALIFYING_FORMULATIONS:
        if re.search(rf"\b{re.escape(term)}\b", target_text):
            continue
        if re.search(rf"\b{re.escape(term)}\b", detected_text):
            return (
                "formulation_mismatch",
                f"Image indicates {term!r} formulation, which does not match target product",
            )
        if term in _STRICT_URL_FORMULATIONS and re.search(
            rf"(?:^|[/_\-]){re.escape(term)}(?:[/_\-]|$)", page_path
        ):
            return (
                "formulation_mismatch",
                f"Source URL path indicates {term!r} formulation, which does not match target product",
            )

    allowed_counts = parse_allowed_pack_counts(gb_pack_size)
    if allowed_counts:
        found_count = check.detected_pack_count or extract_pack_count_from_snippets(
            check.visible_text_snippets
        )
        if found_count is not None and found_count not in allowed_counts:
            return (
                "pack_size_mismatch",
                f"Image shows {found_count} units; Greenbook pack_size allows {sorted(allowed_counts)}",
            )

    return None


def _derive_product_key(bundle: DiscoveryBundle) -> str:
    seed = bundle.enriched_seed
    return seed.nafdac_number or seed.product_name or seed.gtin or "unknown"


def verify_images(
    bundle: DiscoveryBundle,
    provider: LLMProvider,
    storage: ImageStorage,
    *,
    product_key: Optional[str] = None,
    max_unique: int = 6,
    trusted_store=None,
    trusted_only_images: bool = True,
    downloader: Optional[Callable[[str], Tuple[bytes, str]]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    funnel_out: Optional[Dict[str, int]] = None,
) -> Tuple[List[VerifiedReferenceImage], List[Dict], Dict[str, str]]:
    """Download, deduplicate, visually verify via multimodal LLM, and persist valid packaging images."""
    funnel: Dict[str, int] = {}
    candidates = collect_candidate_image_urls(
        bundle,
        trusted_store=trusted_store,
        trusted_only=trusted_only_images,
        funnel_out=funnel,
    )
    if not candidates:
        funnel.update({"downloaded": 0, "passed_vision": 0, "saved": 0})
        if funnel_out is not None:
            funnel_out.update(funnel)
        if progress_cb:
            progress_cb(
                f"[3b] Image funnel: raw={funnel.get('raw_total', 0)} "
                f"(pages={funnel.get('raw_from_pages', 0)}, search={funnel.get('raw_from_search', 0)}) "
                f"-> dropped_alert={funnel.get('dropped_disqualified', 0)}, "
                f"dropped_untrusted={funnel.get('dropped_untrusted', 0)} -> trusted=0 -> saved=0"
            )
        return [], [], {}

    if progress_cb:
        progress_cb(f"[3b] Downloading & deduplicating {len(candidates)} candidate image(s)...")
    t_dl = time.monotonic()
    kept, rejected = download_and_deduplicate_images(
        candidates,
        max_unique=max_unique,
        downloader=downloader,
    )
    if progress_cb:
        progress_cb(
            f"[3b] Image download done in {time.monotonic() - t_dl:.1f}s "
            f"({len(kept)} unique kept, {len(rejected)} pre-filtered)"
        )
    if not kept:
        funnel.update({"downloaded": 0, "passed_vision": 0, "saved": 0})
        if funnel_out is not None:
            funnel_out.update(funnel)
        if progress_cb:
            progress_cb(
                f"[3b] Image funnel: raw={funnel.get('raw_total', 0)} "
                f"(pages={funnel.get('raw_from_pages', 0)}, search={funnel.get('raw_from_search', 0)}) "
                f"-> trusted={funnel.get('after_trusted', 0)} -> downloaded=0 -> saved=0"
            )
        return [], rejected, {}

    prompt = build_image_verify_prompt(bundle, kept)
    image_parts = [prepare_image_for_llm(img.raw_bytes, img.mime_type) for img in kept]
    if progress_cb:
        progress_cb(f"[3c] Verifying {len(image_parts)} unique image(s) with {provider.model_name}...")
    t_vis = time.monotonic()
    batch_res = provider.generate_structured_with_images(
        prompt,
        image_parts,
        BatchImageVerificationResponse,
        system_instruction=IMAGE_VERIFY_SYSTEM_INSTRUCTION,
    )
    if progress_cb:
        progress_cb(f"[3c] Visual verification done in {time.monotonic() - t_vis:.1f}s")

    checks_by_idx = {item.candidate_index: item for item in batch_res.results}
    key = product_key or _derive_product_key(bundle)

    # First pass: evaluate hard rejections (LLM flags + deterministic Greenbook gates)
    surviving: List[Tuple[DownloadedImage, object]] = []
    for idx, img in enumerate(kept):
        check = checks_by_idx.get(idx)
        if check is None:
            rejected.append({"url": img.url, "reason": "llm_omitted"})
            continue

        if not check.is_product_packaging:
            rejected.append(
                {
                    "url": img.url,
                    "reason": "not_product_packaging",
                    "detail": check.rejection_reason,
                }
            )
            continue
        if not check.matches_target_product:
            rejected.append(
                {
                    "url": img.url,
                    "reason": "wrong_product",
                    "detail": check.rejection_reason,
                }
            )
            continue
        if not check.quality_ok:
            rejected.append(
                {
                    "url": img.url,
                    "reason": "low_quality",
                    "detail": check.rejection_reason,
                }
            )
            continue

        det_rejection = _deterministic_image_rejection(check, img, bundle)
        if det_rejection is not None:
            reason, detail = det_rejection
            rejected.append({"url": img.url, "reason": reason, "detail": detail})
            continue

        surviving.append((img, check))

    has_real_photo = any(check.is_real_photo for _img, check in surviving)

    verified: List[VerifiedReferenceImage] = []
    edition_summaries: Dict[str, str] = {}

    for img, check in surviving:
        if has_real_photo and not check.is_real_photo:
            rejected.append(
                {
                    "url": img.url,
                    "reason": "stock_render",
                    "detail": check.rejection_reason
                    or "3D render/mockup dropped because real packaging photos exist",
                }
            )
            continue

        edition_id = (check.packaging_edition_id or "edition_1").strip() or "edition_1"
        pack_variant = (check.pack_size_variant or "").strip() or None
        if not pack_variant and check.detected_pack_count:
            pack_variant = f"{check.detected_pack_count} units"

        bucket_key = f"{pack_variant or 'default'}::{edition_id}"
        if check.edition_summary and bucket_key not in edition_summaries:
            edition_summaries[bucket_key] = check.edition_summary

        storage_path = storage.save(key, img.sha256, img.raw_bytes, img.mime_type)
        verified.append(
            VerifiedReferenceImage(
                image_id=img.sha256[:16],
                storage_path=storage_path,
                source_url=img.url,
                source_page_url=img.source_page_url,
                source_trust_tier=img.source_trust_tier,
                sha256=img.sha256,
                dhash=img.dhash,
                width=img.width,
                height=img.height,
                mime_type=img.mime_type,
                view_type=check.view_type,
                packaging_edition_id=edition_id,
                pack_size_variant=pack_variant,
                is_real_photo=check.is_real_photo,
                has_local_regulatory_markers=check.has_local_regulatory_markers,
                visible_cues=list(check.visible_cues),
                visible_text_snippets=list(check.visible_text_snippets),
                captured_from_domain=img.captured_from_domain,
                captured_from_type=img.captured_from_type,
                captured_at=img.captured_at,
                market_verified=bool(check.has_local_regulatory_markers),
                status="staging",
            )
        )

    funnel.update(
        {
            "downloaded": len(kept),
            "passed_vision": len(surviving),
            "saved": len(verified),
        }
    )
    if funnel_out is not None:
        funnel_out.update(funnel)
    if progress_cb:
        progress_cb(
            f"[3c] Image funnel: raw={funnel.get('raw_total', 0)} "
            f"(pages={funnel.get('raw_from_pages', 0)}, search={funnel.get('raw_from_search', 0)}) "
            f"-> trusted={funnel.get('after_trusted', 0)} -> downloaded={len(kept)} "
            f"-> passed_vision={len(surviving)} -> saved={len(verified)}"
        )

    return verified, rejected, edition_summaries


def group_packaging_versions(
    images: List[VerifiedReferenceImage],
    edition_summaries: Optional[Dict[str, str]] = None,
) -> List[PackagingVersion]:
    """Group verified images by (pack_size_variant, packaging_edition_id) and mark the strongest primary."""
    if not images:
        return []
    summaries = edition_summaries or {}
    buckets: Dict[tuple[str, str], List[VerifiedReferenceImage]] = {}
    for img in images:
        variant_key = (img.pack_size_variant or "").strip().lower()
        edition_key = img.packaging_edition_id or "edition_1"
        buckets.setdefault((variant_key, edition_key), []).append(img)

    versions: List[PackagingVersion] = []
    for (variant_key, edition_id), imgs in buckets.items():
        has_reg = any(i.has_local_regulatory_markers for i in imgs)
        display_variant = next((i.pack_size_variant for i in imgs if i.pack_size_variant), None)
        summary_lookup_key = f"{display_variant or 'default'}::{edition_id}"
        versions.append(
            PackagingVersion(
                edition_id=edition_id,
                pack_size_variant=display_variant,
                summary=summaries.get(summary_lookup_key) or summaries.get(edition_id),
                is_primary=False,
                has_local_regulatory_markers=has_reg,
                status="staging",
                images=imgs,
            )
        )

    def _rank_key(v: PackagingVersion) -> tuple:
        best_res = max((i.width * i.height for i in v.images), default=0)
        best_tier = min((i.source_trust_tier for i in v.images), default=3)
        return (
            not v.has_local_regulatory_markers,
            best_tier,
            -len(v.images),
            -best_res,
            v.edition_id,
        )

    versions.sort(key=_rank_key)
    versions[0].is_primary = True
    return versions


def synthesize_dossier(
    bundle: DiscoveryBundle,
    *,
    provider: Optional[LLMProvider] = None,
    storage: Optional[ImageStorage] = None,
    product_key: Optional[str] = None,
    verify_visuals: bool = True,
    max_unique_images: int = 6,
    trusted_store=None,
    trusted_only_images: bool = True,
    image_downloader: Optional[Callable[[str], Tuple[bytes, str]]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> CandidateProductDossier:
    """Synthesize a source-grounded CandidateProductDossier from a DiscoveryBundle."""
    t0 = time.monotonic()
    active_provider = provider or get_default_provider()
    active_storage = storage or LocalImageStorage()

    dossier_prompt = build_dossier_prompt(bundle)
    if progress_cb:
        progress_cb(f"[3a] Synthesizing dossier with {active_provider.model_name}...")
    t_dos = time.monotonic()
    extracted = active_provider.generate_structured(
        dossier_prompt,
        ExtractedDossierResponse,
        system_instruction=DOSSIER_SYSTEM_INSTRUCTION,
    )
    if progress_cb:
        progress_cb(
            f"[3a] Dossier synthesized in {time.monotonic() - t_dos:.1f}s "
            f"(model={active_provider.model_name})"
        )

    verified_images: List[VerifiedReferenceImage] = []
    rejected_images: List[Dict] = []
    packaging_versions: List[PackagingVersion] = []
    image_funnel: Dict[str, int] = {}
    if verify_visuals:
        verified_images, rejected_images, edition_summaries = verify_images(
            bundle,
            active_provider,
            active_storage,
            product_key=product_key,
            max_unique=max_unique_images,
            trusted_store=trusted_store,
            trusted_only_images=trusted_only_images,
            downloader=image_downloader,
            progress_cb=progress_cb,
            funnel_out=image_funnel,
        )
        packaging_versions = group_packaging_versions(verified_images, edition_summaries)

    merged_gtins = collect_validated_gtins(bundle)
    for g in extracted.gtins:
        clean_g = re.sub(r"\D", "", str(g or ""))
        if len(clean_g) in (8, 12, 13, 14) and clean_g not in merged_gtins:
            merged_gtins.append(clean_g)

    shelf_life = extracted.shelf_life_months
    if shelf_life is None:
        for src in bundle.sources:
            if src.is_smpc_pdf and src.content:
                sec = extract_smpc_key_sections(src.content)
                shelf_life = parse_shelf_life_months(sec)
                if shelf_life is not None:
                    break

    normalized_tells = _normalize_counterfeit_tells(list(extracted.known_counterfeit_tells))
    ref_level = compute_reference_level(packaging_versions)

    elapsed = round(time.monotonic() - t0, 3)
    dossier = CandidateProductDossier(
        seed=bundle.seed.to_dict(),
        enriched_seed=bundle.enriched_seed.to_dict(),
        brand_name=extracted.brand_name,
        generic_name=extracted.generic_name,
        active_ingredients=list(extracted.active_ingredients),
        strength=extracted.strength,
        dosage_form=extracted.dosage_form,
        pack_size=extracted.pack_size,
        manufacturer=extracted.manufacturer,
        nafdac_number=extracted.nafdac_number,
        registration_status=extracted.registration_status,
        gtins=merged_gtins,
        shelf_life_months=shelf_life,
        storage_statement=extracted.storage_statement,
        packaging_description=extracted.packaging_description,
        color_scheme=list(extracted.color_scheme),
        distinctive_features=list(extracted.distinctive_features),
        expected_security_features=list(extracted.expected_security_features),
        known_counterfeit_tells=normalized_tells,
        recall_alerts=list(extracted.recall_alerts),
        version_notes=list(extracted.version_notes),
        claims=list(extracted.claims),
        reference_level=ref_level,
        packaging_versions=packaging_versions,
        reference_images=verified_images,
        rejected_images=rejected_images,
        model_used=active_provider.model_name,
        stats={
            "elapsed_s": elapsed,
            "claims_count": len(extracted.claims),
            "gtins_count": len(merged_gtins),
            "counterfeit_tells_count": len(normalized_tells),
            "security_features_count": len(extracted.expected_security_features),
            "reference_level": ref_level,
            "packaging_versions_count": len(packaging_versions),
            "reference_images_count": len(verified_images),
            "rejected_images_count": len(rejected_images),
            "image_funnel": image_funnel,
            "sources_used": len(bundle.sources),
            "greenbook_matches_used": len(bundle.greenbook_matches),
        },
    )

    from validation.evidence import validate_dossier_evidence  # noqa: PLC0415

    validated_dossier, gate_report = validate_dossier_evidence(bundle, dossier, sanitize=True)
    validated_dossier.stats["evidence_gate"] = gate_report.model_dump()
    validated_dossier.stats["dossier_status"] = gate_report.dossier_status
    return validated_dossier
