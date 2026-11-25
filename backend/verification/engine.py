"""GENIE AI Multi-Signal Verification Engine with Coverage-Based Graceful Degradation (Phase D).

Core invariants enforced in code:
  1. Hard negatives fire at ANY coverage (`likely_counterfeit` > `suspicious`).
  2. Missing signals never count as negatives — they lower coverage and cap positive ceilings.
  3. NAFDAC-derived signals can disqualify a pack, never confirm one on their own:
     reaching `likely_authentic` requires positive visual matching or an externally verified
     security feature; reaching `authentic` requires Level A/B approved visual matching and
     coverage >= 0.65.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from registry.greenbook import GreenbookStore, default_db_path, normalize_nafdac_number
from scraper.extract.structured import extract_nafdac_numbers

from .schemas import (
    ScanInput,
    SignalResult,
    VerificationResult,
    VerificationVerdict,
    to_ml_infra_verdict,
)
from .signals import (
    evaluate_batch_expiry_signal,
    evaluate_gtin_signal,
    evaluate_nafdac_signal,
    evaluate_recalls_and_tells_signal,
    evaluate_security_features_signal,
    evaluate_spec_ocr_signal,
    evaluate_visual_signal,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRODUCTS_DIR = REPO_ROOT / "data" / "products"

AUTHENTIC_MIN_COVERAGE = 0.65
UNKNOWN_MIN_COVERAGE = 0.30


def load_saved_dossiers(products_dir: Path = DEFAULT_PRODUCTS_DIR) -> List[Dict[str, Any]]:
    """Load all saved `data/products/<slug>/product.json` dossiers from disk."""
    base = Path(products_dir)
    if not base.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for p_json in sorted(base.glob("*/product.json")):
        try:
            data = json.loads(p_json.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("_slug", p_json.parent.name)
                out.append(data)
        except (OSError, ValueError):
            continue
    return out


def _dossier_from_greenbook_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Build an in-memory Level-C baseline dossier directly from a NAFDAC Greenbook record."""
    ingredients = [
        p.strip()
        for p in str(row.get("ingredient_name") or "").split(";")
        if p.strip()
    ]
    return {
        "_from_greenbook_only": True,
        "brand_name": row.get("product_name"),
        "generic_name": row.get("ingredient_name") or None,
        "active_ingredients": ingredients,
        "strength": row.get("strength") or None,
        "dosage_form": row.get("form_name") or None,
        "pack_size": row.get("pack_size") or None,
        "manufacturer": row.get("applicant_name") or None,
        "nafdac_number": row.get("nafdac_number") or None,
        "registration_status": row.get("status") or None,
        "gtins": [],
        "shelf_life_months": None,
        "storage_statement": None,
        "packaging_description": row.get("product_description") or None,
        "expected_security_features": [],
        "known_counterfeit_tells": [],
        "recall_alerts": [],
        "packaging_versions": [],
        "reference_level": "C",
    }


def _clean_digits(raw: Optional[str]) -> str:
    return re.sub(r"\D", "", str(raw or ""))


def _brand_tokens(text: Optional[str]) -> List[str]:
    stop = {"mg", "mcg", "ml", "tablet", "tablets", "capsule", "capsules", "caplet", "caplets", "syrup", "suspension", "oral", "pack"}
    return [
        t.lower()
        for t in re.findall(r"[A-Za-z]+", str(text or ""))
        if len(t) >= 3 and t.lower() not in stop
    ]


def _resolve_product(
    scan: ScanInput,
    dossiers: List[Dict[str, Any]],
    greenbook: Optional[GreenbookStore],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Resolve canonical product dossier from saved dossiers first, then local Greenbook.

    Returns `(dossier_or_none, recommend_discovery)`.
    """
    raw_nafdac = (scan.nafdac_number or "").strip()
    if not raw_nafdac and scan.ocr_text:
        found = extract_nafdac_numbers(scan.ocr_text)
        if found:
            raw_nafdac = found[0]
    norm_nafdac = normalize_nafdac_number(raw_nafdac)
    clean_gtin = _clean_digits(scan.gtin)

    # 1. Exact NAFDAC match in saved dossiers
    if norm_nafdac:
        for d in dossiers:
            if normalize_nafdac_number(d.get("nafdac_number") or "") == norm_nafdac:
                return d, False

    # 2. Exact GTIN match in saved dossiers
    if clean_gtin:
        for d in dossiers:
            d_gtins = {_clean_digits(g) for g in (d.get("gtins") or [])}
            if clean_gtin in d_gtins:
                return d, False

    # 3. Product name hint match in saved dossiers
    hint = (scan.product_name_hint or "").strip()
    if hint:
        hint_low = hint.lower()
        hint_tokens = set(_brand_tokens(hint))
        best_match: Optional[Dict[str, Any]] = None
        best_overlap = 0
        for d in dossiers:
            brand = str(d.get("brand_name") or "").strip()
            if not brand:
                continue
            brand_low = brand.lower()
            if hint_low == brand_low or hint_low in brand_low or brand_low in hint_low:
                return d, False
            overlap = len(hint_tokens & set(_brand_tokens(brand)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = d
        if best_match is not None and best_overlap >= 1:
            return best_match, False

    # 4. Fallback to local 9,065-drug NAFDAC Greenbook mirror (degraded Level C, recommends discovery)
    if greenbook is not None:
        if norm_nafdac:
            rows = greenbook.lookup_by_nafdac(norm_nafdac)
            if rows:
                return _dossier_from_greenbook_row(rows[0]), True
        if hint:
            hint_brand = set(_brand_tokens(hint))
            rows = greenbook.search(hint, limit=3)
            for row in rows:
                row_brand = set(_brand_tokens(row.get("product_name")))
                if not hint_brand or (hint_brand & row_brand):
                    return _dossier_from_greenbook_row(row), True

    return None, True


_UNSET: Any = object()


class VerificationEngine:
    """Orchestrates all 7 verification signals and enforces graceful degradation rules."""

    def __init__(
        self,
        *,
        products_dir: Path = DEFAULT_PRODUCTS_DIR,
        dossiers: Optional[List[Dict[str, Any]]] = None,
        greenbook: Any = _UNSET,
        cv_adapter: Optional[Any] = None,
    ) -> None:
        self._products_dir = Path(products_dir)
        self._explicit_dossiers = dossiers
        if greenbook is _UNSET:
            gb_path = default_db_path()
            self._greenbook: Optional[GreenbookStore] = (
                GreenbookStore(gb_path) if gb_path.is_file() else None
            )
        else:
            self._greenbook = greenbook
        self._cv_adapter = cv_adapter

    def _get_dossiers(self) -> List[Dict[str, Any]]:
        if self._explicit_dossiers is not None:
            return self._explicit_dossiers
        return load_saved_dossiers(self._products_dir)

    def verify(self, scan: ScanInput) -> VerificationResult:
        """Verify a consumer scan against curated dossiers + NAFDAC Greenbook."""
        all_dossiers = self._get_dossiers()
        dossier, recommend_discovery = _resolve_product(scan, all_dossiers, self._greenbook)

        # Unregistered / unknown product handling
        if dossier is None:
            has_identifier = bool(
                (scan.nafdac_number or "").strip()
                or (scan.product_name_hint or "").strip()
                or (scan.gtin or "").strip()
            )
            if has_identifier:
                verdict: VerificationVerdict = "suspicious"
                summary = (
                    "Product was not found in the NAFDAC Greenbook registry or GENIE dossiers "
                    "(unregistered product finding — autonomous discovery recommended)."
                )
                return VerificationResult(
                    verdict=verdict,
                    ml_infra_verdict=to_ml_infra_verdict(verdict),
                    confidence=0.75,
                    coverage=0.0,
                    reference_level="C",
                    matched_product=None,
                    matched_version=None,
                    unregistered_product=True,
                    recommend_discovery=True,
                    signals=[],
                    limitations=[
                        "Product is not registered in the local NAFDAC Greenbook mirror; no reference specs or photos exist.",
                    ],
                    counterfeit_tell_hits=[],
                    summary=summary,
                )

            verdict = "unknown"
            return VerificationResult(
                verdict=verdict,
                ml_infra_verdict=to_ml_infra_verdict(verdict),
                confidence=0.0,
                coverage=0.0,
                reference_level="C",
                unregistered_product=False,
                recommend_discovery=True,
                limitations=["No product name, NAFDAC number, or barcode was supplied to identify the product."],
                summary="Could not identify the product from this scan.",
            )

        # Run all 7 signals
        sig_visual, ref_level, matched_version = evaluate_visual_signal(
            scan, dossier, cv_adapter=self._cv_adapter
        )
        sig_nafdac = evaluate_nafdac_signal(scan, dossier, greenbook=self._greenbook)
        sig_spec = evaluate_spec_ocr_signal(scan, dossier)
        sig_batch = evaluate_batch_expiry_signal(scan, dossier)
        sig_gtin = evaluate_gtin_signal(scan, dossier, all_dossiers=all_dossiers)
        sig_sec = evaluate_security_features_signal(scan, dossier)
        sig_recalls, tell_hits = evaluate_recalls_and_tells_signal(scan, dossier)

        signals: List[SignalResult] = [
            sig_visual,
            sig_nafdac,
            sig_spec,
            sig_batch,
            sig_gtin,
            sig_sec,
            sig_recalls,
        ]

        # Coverage across applicable scoring signals (excludes not_applicable and 0-weight safety gates)
        applicable_weight = sum(
            s.max_weight for s in signals if s.status != "not_applicable" and s.max_weight > 0
        )
        available_weight = sum(
            s.weight for s in signals if s.status == "available" and s.weight > 0
        )
        coverage = round(available_weight / applicable_weight, 3) if applicable_weight > 0 else 0.0

        # Auto-collect limitations
        limitations: List[str] = []
        if dossier.get("_from_greenbook_only"):
            limitations.append(
                "Only baseline NAFDAC Greenbook data is cached for this product — full GENIE discovery dossier has not run yet."
            )
        for s in signals:
            if s.limitation and s.limitation not in limitations:
                limitations.append(s.limitation)

        matched_product = {
            "brand_name": dossier.get("brand_name"),
            "generic_name": dossier.get("generic_name"),
            "strength": dossier.get("strength"),
            "dosage_form": dossier.get("dosage_form"),
            "pack_size": dossier.get("pack_size"),
            "manufacturer": dossier.get("manufacturer"),
            "nafdac_number": dossier.get("nafdac_number"),
            "registration_status": dossier.get("registration_status"),
        }

        # Step 1: Hard negatives apply at ANY coverage (`likely_counterfeit` > `suspicious`)
        hard_negatives = [s for s in signals if s.hard_negative is not None]
        if hard_negatives:
            is_likely_fake = any(s.hard_negative == "likely_counterfeit" for s in hard_negatives)
            verdict = "likely_counterfeit" if is_likely_fake else "suspicious"
            worst_strength = max(s.strength for s in hard_negatives)
            confidence = round(max(0.85 if is_likely_fake else 0.75, worst_strength), 3)
            reasons = "; ".join(s.summary for s in hard_negatives)
            return VerificationResult(
                verdict=verdict,
                ml_infra_verdict=to_ml_infra_verdict(verdict),
                confidence=confidence,
                coverage=coverage,
                reference_level=ref_level,  # type: ignore[arg-type]
                matched_product=matched_product,
                matched_version=matched_version,
                unregistered_product=False,
                recommend_discovery=recommend_discovery,
                signals=signals,
                limitations=limitations,
                counterfeit_tell_hits=tell_hits,
                summary=f"Flagged as {verdict.replace('_', ' ')}: {reasons}",
            )

        # Step 2: Weighted score over available scoring signals only (missing signals never penalize)
        scoring_available = [s for s in signals if s.status == "available" and s.weight > 0]
        if scoring_available:
            denom = sum(s.weight for s in scoring_available)
            weighted_sum = 0.0
            for s in scoring_available:
                if s.direction == "positive":
                    weighted_sum += s.weight * s.strength
                elif s.direction == "negative":
                    weighted_sum -= s.weight * s.strength
            score = max(0.0, min(1.0, weighted_sum / denom)) if denom > 0 else 0.0
        else:
            score = 0.0

        has_soft_negative = any(
            s.status == "available" and s.direction == "negative" for s in signals
        )
        has_positive_visual = (
            sig_visual.status == "available"
            and sig_visual.direction == "positive"
            and sig_visual.strength >= 0.75
        )
        has_unlocking_independent = any(
            s.status == "available"
            and s.direction == "positive"
            and s.unlocks_likely_authentic
            for s in signals
        )

        # Step 3: Strict positive verdict ceilings (NAFDAC alone can NEVER confirm authenticity)
        if has_soft_negative:
            verdict = "suspicious" if score < 0.35 else "unknown"
            confidence = round(max(0.25, 1.0 - score), 3)
            summary = "Verification inconclusive due to a low-confidence mismatch — retake a clearer scan."
        elif (
            ref_level in ("A", "B")
            and has_positive_visual
            and coverage >= AUTHENTIC_MIN_COVERAGE
            and score >= 0.75
        ):
            verdict = "authentic"
            confidence = round(min(0.99, score * (0.7 + 0.3 * coverage)), 3)
            summary = (
                f"Verified authentic against Level {ref_level} approved reference "
                f"(coverage {coverage:.0%})."
            )
        elif has_unlocking_independent and score >= 0.55:
            verdict = "likely_authentic"
            confidence = round(min(0.88, score * (0.65 + 0.35 * coverage)), 3)
            summary = (
                f"Likely authentic based on independent verification "
                f"(coverage {coverage:.0%}; review limitations for unchecked layers)."
            )
        elif coverage >= UNKNOWN_MIN_COVERAGE and score >= 0.50:
            verdict = "registered_unverified"
            confidence = round(min(0.60, score * coverage), 3)
            summary = (
                "Registered & spec-consistent — packaging was not independently verified "
                "(NAFDAC data alone cannot confirm authenticity)."
            )
        else:
            verdict = "unknown"
            confidence = round(score * coverage, 3)
            summary = (
                f"Could not verify (coverage {coverage:.0%}) — insufficient signals available."
            )

        return VerificationResult(
            verdict=verdict,
            ml_infra_verdict=to_ml_infra_verdict(verdict),
            confidence=confidence,
            coverage=coverage,
            reference_level=ref_level,  # type: ignore[arg-type]
            matched_product=matched_product,
            matched_version=matched_version,
            unregistered_product=False,
            recommend_discovery=recommend_discovery,
            signals=signals,
            limitations=limitations,
            counterfeit_tell_hits=tell_hits,
            summary=summary,
        )
