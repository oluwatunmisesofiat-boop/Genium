"""Deterministic Evidence Gate: verbatim quote verification, numeric grounding, and Tier 1 conflict detection.

Protects downstream verification from LLM hallucinations by ensuring:
  1. Verbatim quotes actually appear in their cited source text (whitespace & curly-quote normalized).
  2. High-stakes numeric/identifier claims (NAFDAC number, GTINs, SmPC shelf_life_months,
     compromised batch numbers) are literally grounded in source text; ungrounded batch numbers,
     GTINs, or shelf_life values are quarantined before they can cause false counterfeit verdicts.
  3. Any Tier 1 Greenbook conflict blocks auto-promotion (`staging` -> `approved`).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from discovery.models import DiscoveryBundle
from intelligence.schemas import CandidateProductDossier, CounterfeitTell
from registry.greenbook import normalize_nafdac_number

_WS_RE = re.compile(r"\s+")
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


def normalize_text_for_match(text: str) -> str:
    """Normalize whitespace, smart quotes, dashes, and NBSP for reliable substring verification."""
    return _WS_RE.sub(" ", (text or "").translate(_TYPO_TRANS)).strip()


def _format_greenbook_record(match: Dict[str, Any]) -> str:
    source_url = str(match.get("detail_url") or "")
    lines = [
        "### NAFDAC Greenbook Record (Tier 1 Regulatory Mirror)",
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


def build_source_corpus(bundle: DiscoveryBundle) -> Dict[str, str]:
    """Build a normalized `{source_url: text}` lookup covering Greenbook matches and fetched sources."""
    corpus: Dict[str, str] = {}
    for match in bundle.greenbook_matches or []:
        url = str(match.get("detail_url") or "").strip()
        if url:
            corpus[url] = normalize_text_for_match(_format_greenbook_record(match))

    for src in bundle.sources or []:
        parts = [src.title or "", src.content or ""]
        if src.product:
            parts.append(json.dumps(src.product, ensure_ascii=False))
        if src.identifiers:
            parts.append(json.dumps(src.identifiers, ensure_ascii=False))
        corpus[src.url] = normalize_text_for_match("\n".join(parts))

    return corpus


def _quote_in_text(quote: str, source_text: str) -> bool:
    q = normalize_text_for_match(quote)
    t = normalize_text_for_match(source_text)
    if not q or not t:
        return False
    return q in t or q.lower() in t.lower()


def _shelf_life_grounded(months: int, full_text: str) -> bool:
    low = full_text.lower()
    if re.search(rf"\b{months}\s*(?:months?|mos?)\b", low):
        return True
    if months % 12 == 0:
        years = months // 12
        if re.search(rf"\b{years}\s*(?:years?|yrs?)\b", low):
            return True
    else:
        years_f = months / 12.0
        yr_str = f"{years_f:.1f}".rstrip("0").rstrip(".")
        if re.search(rf"\b{re.escape(yr_str)}\s*(?:years?|yrs?)\b", low):
            return True
    return False


def _token_overlap(a: str, b: str) -> bool:
    stop = {"limited", "ltd", "nigeria", "plc", "pharma", "pharmaceutical", "consumer", "company", "inc"}
    ta = {t.lower() for t in re.findall(r"[A-Za-z]+", a or "") if len(t) >= 3 and t.lower() not in stop}
    tb = {t.lower() for t in re.findall(r"[A-Za-z]+", b or "") if len(t) >= 3 and t.lower() not in stop}
    if not ta or not tb:
        return True
    return bool(ta & tb)


class EvidenceGateReport(BaseModel):
    dossier_status: Literal["staging", "approved"] = "staging"
    total_claims: int = 0
    matched_claims: int = 0
    unmatched_claims: List[Dict[str, Any]] = Field(default_factory=list)
    numeric_failures: List[str] = Field(default_factory=list)
    quarantined_items: List[str] = Field(default_factory=list)
    tier1_conflicts: List[str] = Field(default_factory=list)
    blocking_reasons: List[str] = Field(default_factory=list)


def validate_dossier_evidence(
    bundle: DiscoveryBundle,
    dossier: CandidateProductDossier,
    *,
    sanitize: bool = True,
) -> Tuple[CandidateProductDossier, EvidenceGateReport]:
    """Verify verbatim quotes, ground critical numeric claims, detect Tier 1 conflicts, and gate promotion."""
    corpus = build_source_corpus(bundle)
    full_corpus_text = "\n".join(corpus.values())

    # 1. Verbatim quote check on EvidenceClaims
    matched = 0
    unmatched: List[Dict[str, Any]] = []
    for claim in dossier.claims:
        src_text = corpus.get(claim.source_url, "")
        if _quote_in_text(claim.verbatim_quote, src_text):
            matched += 1
        else:
            unmatched.append(
                {
                    "field_name": claim.field_name,
                    "value": claim.value,
                    "source_url": claim.source_url,
                    "verbatim_quote": claim.verbatim_quote,
                    "source_found": bool(src_text),
                }
            )

    numeric_failures: List[str] = []
    quarantined: List[str] = []
    updates: Dict[str, Any] = {}

    # 2a. Ground NAFDAC number
    if dossier.nafdac_number:
        norm_nafdac = normalize_nafdac_number(dossier.nafdac_number)
        norm_corpus = normalize_nafdac_number(full_corpus_text)
        if norm_nafdac and norm_nafdac not in norm_corpus:
            numeric_failures.append(f"nafdac_number {dossier.nafdac_number!r} not found verbatim in any source")

    # 2b. Ground GTINs (drop any hallucinated GTIN not present in sources or seed)
    seed_gtin = re.sub(r"\D", "", str(bundle.enriched_seed.gtin or ""))
    corpus_digits = re.sub(r"\D", "", full_corpus_text)
    clean_gtins: List[str] = []
    for g in dossier.gtins:
        gd = re.sub(r"\D", "", str(g or ""))
        if gd and (gd == seed_gtin or gd in corpus_digits):
            clean_gtins.append(gd)
        else:
            numeric_failures.append(f"gtin {g!r} not found in source corpus")
            quarantined.append(f"gtin:{g}")
    if sanitize and len(clean_gtins) != len(dossier.gtins):
        updates["gtins"] = clean_gtins

    # 2c. Ground shelf_life_months (prevent hallucinated shelf life from triggering false suspicious verdicts)
    if dossier.shelf_life_months is not None:
        if not _shelf_life_grounded(dossier.shelf_life_months, full_corpus_text):
            numeric_failures.append(
                f"shelf_life_months={dossier.shelf_life_months} not grounded in SmPC or scraped sources"
            )
            quarantined.append(f"shelf_life_months:{dossier.shelf_life_months}")
            if sanitize:
                updates["shelf_life_months"] = None

    # 2d. Ground compromised batch numbers in known_counterfeit_tells
    clean_tells: List[CounterfeitTell] = []
    for tell in dossier.known_counterfeit_tells:
        src_text = corpus.get(tell.source_url or "", "") or full_corpus_text
        if tell.affected_batches:
            grounded_batches = []
            for batch in tell.affected_batches:
                pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(batch)}(?![A-Za-z0-9])", re.IGNORECASE)
                if pat.search(src_text):
                    grounded_batches.append(batch)
                else:
                    numeric_failures.append(
                        f"counterfeit tell batch {batch!r} not found in cited source"
                    )
                    quarantined.append(f"batch:{batch}")
            if grounded_batches:
                clean_tells.append(tell.model_copy(update={"affected_batches": grounded_batches}))
            elif tell.tell_type != "compromised_batch":
                clean_tells.append(tell.model_copy(update={"affected_batches": []}))
            else:
                quarantined.append(f"tell:{tell.summary[:40]}")
        else:
            if tell.verbatim_quote and tell.source_url and not _quote_in_text(tell.verbatim_quote, src_text):
                quarantined.append(f"tell_ungrounded_quote:{tell.summary[:40]}")
                continue
            clean_tells.append(tell)
    if sanitize and len(clean_tells) != len(dossier.known_counterfeit_tells) or any(
        a.affected_batches != b.affected_batches
        for a, b in zip(clean_tells, dossier.known_counterfeit_tells)
    ):
        updates["known_counterfeit_tells"] = clean_tells

    # 3. Tier 1 Greenbook conflict detection
    tier1_conflicts: List[str] = []
    has_tier1_anchor = bool(bundle.greenbook_matches) or any(
        s.trust_tier == 1 for s in (bundle.sources or [])
    )
    if bundle.greenbook_matches:
        gb = bundle.greenbook_matches[0]
        gb_nafdac = normalize_nafdac_number(gb.get("nafdac_number") or "")
        dos_nafdac = normalize_nafdac_number(dossier.nafdac_number or "")
        if gb_nafdac and dos_nafdac and gb_nafdac != dos_nafdac:
            tier1_conflicts.append(
                f"NAFDAC mismatch vs Tier 1 Greenbook: dossier={dos_nafdac}, greenbook={gb_nafdac}"
            )
        gb_mfr = str(gb.get("applicant_name") or "").strip()
        dos_mfr = str(dossier.manufacturer or "").strip()
        if gb_mfr and dos_mfr and not _token_overlap(gb_mfr, dos_mfr):
            tier1_conflicts.append(
                f"Manufacturer mismatch vs Tier 1 Greenbook: dossier={dos_mfr!r}, greenbook={gb_mfr!r}"
            )

    # 4. Promotion decision (`staging` -> `approved` for text dossier only; images stay gated separately)
    blocking: List[str] = []
    if not has_tier1_anchor:
        blocking.append("No Tier 1 regulatory/manufacturer anchor source")
    if tier1_conflicts:
        blocking.append(f"{len(tier1_conflicts)} Tier 1 conflict(s) detected")
    if numeric_failures:
        blocking.append(f"{len(numeric_failures)} numeric grounding failure(s)")
    if dossier.claims:
        ratio = matched / len(dossier.claims)
        if ratio < 0.80:
            blocking.append(f"Verbatim quote match rate {ratio:.0%} below 80% threshold")
    else:
        blocking.append("No verbatim evidence claims emitted")

    status: Literal["staging", "approved"] = "approved" if not blocking else "staging"
    report = EvidenceGateReport(
        dossier_status=status,
        total_claims=len(dossier.claims),
        matched_claims=matched,
        unmatched_claims=unmatched,
        numeric_failures=numeric_failures,
        quarantined_items=quarantined,
        tier1_conflicts=tier1_conflicts,
        blocking_reasons=blocking,
    )

    updates["dossier_status"] = status
    updates["evidence_summary"] = {
        "dossier_status": status,
        "claims_matched": matched,
        "claims_total": len(dossier.claims),
        "numeric_failures": numeric_failures,
        "quarantined_items": quarantined,
        "tier1_conflicts": tier1_conflicts,
        "blocking_reasons": blocking,
    }
    updated_dossier = dossier.model_copy(update=updates)
    return updated_dossier, report
