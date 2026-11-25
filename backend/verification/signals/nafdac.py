"""NAFDAC registration status and identity signal evaluator (`nafdac_derived`, weight=0.20)."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from registry.greenbook import GreenbookStore, normalize_nafdac_number
from scraper.extract.structured import extract_nafdac_numbers

from ..schemas import ScanInput, SignalResult

NAFDAC_WEIGHT = 0.20

_ACTIVE_STATUSES = {"active", "valid", "registered", "approved"}
_RENEWAL_STATUSES = {
    "ongoing renewal",
    "renewal in progress",
    "under renewal",
    "pending renewal",
}
_INACTIVE_STATUSES = {"inactive"}
_EXPIRED_STATUSES = {"expired", "lapsed"}
_REVOKED_STATUSES = {"revoked", "cancelled", "canceled", "suspended", "withdrawn", "banned"}


def _brand_head_token(name: Optional[str]) -> str:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", str(name or "")) if len(t) >= 3]
    return tokens[0] if tokens else ""


def evaluate_nafdac_signal(
    scan: ScanInput,
    dossier: Dict[str, Any],
    *,
    greenbook: Optional[GreenbookStore] = None,
) -> SignalResult:
    """Evaluate NAFDAC registration existence, status mapping, and product match."""
    dossier_nafdac = normalize_nafdac_number(dossier.get("nafdac_number") or "")
    submitted_raw = (scan.nafdac_number or "").strip()
    if not submitted_raw and scan.ocr_text:
        ocr_codes = extract_nafdac_numbers(scan.ocr_text)
        if ocr_codes:
            submitted_raw = ocr_codes[0]
    submitted_nafdac = normalize_nafdac_number(submitted_raw)

    gb_row: Optional[Dict[str, Any]] = None
    if greenbook is not None:
        lookup_code = submitted_nafdac or dossier_nafdac
        if lookup_code:
            rows = greenbook.lookup_by_nafdac(lookup_code)
            if rows:
                gb_row = rows[0]

    # 1. Check submitted NAFDAC code against Greenbook and dossier
    if submitted_nafdac:
        if dossier_nafdac and submitted_nafdac != dossier_nafdac:
            other_name = (gb_row or {}).get("product_name")
            detail = (
                f"Submitted NAFDAC {submitted_nafdac} belongs to {other_name!r}, not {dossier.get('brand_name')!r}"
                if other_name
                else f"Submitted NAFDAC {submitted_nafdac} does not match registered code {dossier_nafdac}"
            )
            return SignalResult(
                name="nafdac",
                source="nafdac_derived",
                status="available",
                direction="negative",
                strength=1.0,
                hard_negative="likely_counterfeit",
                weight=NAFDAC_WEIGHT,
                max_weight=NAFDAC_WEIGHT,
                summary=detail,
                evidence={
                    "submitted_nafdac": submitted_nafdac,
                    "expected_nafdac": dossier_nafdac,
                    "greenbook_product": other_name,
                },
            )
        if greenbook is not None and gb_row is None and not dossier_nafdac:
            return SignalResult(
                name="nafdac",
                source="nafdac_derived",
                status="available",
                direction="negative",
                strength=1.0,
                hard_negative="likely_counterfeit",
                weight=NAFDAC_WEIGHT,
                max_weight=NAFDAC_WEIGHT,
                summary=f"NAFDAC number {submitted_nafdac} was not found in the NAFDAC Greenbook registry.",
                evidence={"submitted_nafdac": submitted_nafdac},
            )
        if gb_row is not None and dossier.get("brand_name"):
            gb_brand = _brand_head_token(gb_row.get("product_name"))
            dos_brand = _brand_head_token(dossier.get("brand_name"))
            if gb_brand and dos_brand and gb_brand != dos_brand:
                return SignalResult(
                    name="nafdac",
                    source="nafdac_derived",
                    status="available",
                    direction="negative",
                    strength=1.0,
                    hard_negative="likely_counterfeit",
                    weight=NAFDAC_WEIGHT,
                    max_weight=NAFDAC_WEIGHT,
                    summary=(
                        f"NAFDAC {submitted_nafdac} is registered to {gb_row.get('product_name')!r}, "
                        f"not {dossier.get('brand_name')!r}."
                    ),
                    evidence={
                        "submitted_nafdac": submitted_nafdac,
                        "greenbook_product": gb_row.get("product_name"),
                    },
                )

    if not submitted_nafdac and not dossier_nafdac:
        return SignalResult(
            name="nafdac",
            source="nafdac_derived",
            status="unavailable",
            direction="neutral",
            strength=0.0,
            weight=0.0,
            max_weight=NAFDAC_WEIGHT,
            summary="No NAFDAC registration number available to check.",
            limitation="NAFDAC registration number was not scanned or found in the dossier.",
        )

    # 2. Evaluate registration status with explicit mapping (Ongoing Renewal != Expired)
    raw_status = str(
        (gb_row or {}).get("status") or dossier.get("registration_status") or ""
    ).strip()
    norm_status = " ".join(raw_status.lower().split())
    code_used = submitted_nafdac or dossier_nafdac
    evidence = {
        "nafdac_number": code_used,
        "registration_status": raw_status or "unknown",
        "code_scanned_on_pack": bool(submitted_nafdac),
        "manufacturer": (gb_row or {}).get("applicant_name") or dossier.get("manufacturer"),
    }

    if norm_status in _REVOKED_STATUSES:
        return SignalResult(
            name="nafdac",
            source="nafdac_derived",
            status="available",
            direction="negative",
            strength=1.0,
            hard_negative="likely_counterfeit",
            weight=NAFDAC_WEIGHT,
            max_weight=NAFDAC_WEIGHT,
            summary=f"NAFDAC registration {code_used} is {raw_status}.",
            evidence=evidence,
        )

    if norm_status in _EXPIRED_STATUSES:
        return SignalResult(
            name="nafdac",
            source="nafdac_derived",
            status="available",
            direction="negative",
            strength=0.8,
            hard_negative="suspicious",
            weight=NAFDAC_WEIGHT,
            max_weight=NAFDAC_WEIGHT,
            summary=(
                f"NAFDAC registration {code_used} is marked Expired "
                "(could be older legal stock or lapsed registration — inspect expiry date carefully)."
            ),
            evidence=evidence,
        )

    if norm_status in _RENEWAL_STATUSES:
        return SignalResult(
            name="nafdac",
            source="nafdac_derived",
            status="available",
            direction="positive",
            strength=0.9 if submitted_nafdac else 0.7,
            weight=NAFDAC_WEIGHT,
            max_weight=NAFDAC_WEIGHT,
            summary=(
                f"NAFDAC {code_used} is registered (status: {raw_status}; treated as active during renewal)."
            ),
            evidence=evidence,
        )

    if norm_status in _INACTIVE_STATUSES:
        return SignalResult(
            name="nafdac",
            source="nafdac_derived",
            status="available",
            direction="neutral",
            strength=0.0,
            weight=NAFDAC_WEIGHT,
            max_weight=NAFDAC_WEIGHT,
            summary=(
                f"NAFDAC {code_used} is listed as 'Inactive' in Greenbook (commonly indicates pending renewal, not revocation)."
            ),
            evidence=evidence,
            limitation="NAFDAC Greenbook lists status as 'Inactive' (often pending renewal) — inspect expiry date and visual markers closely.",
        )

    if norm_status in _ACTIVE_STATUSES:
        return SignalResult(
            name="nafdac",
            source="nafdac_derived",
            status="available",
            direction="positive",
            strength=1.0 if submitted_nafdac else 0.75,
            weight=NAFDAC_WEIGHT,
            max_weight=NAFDAC_WEIGHT,
            summary=f"NAFDAC {code_used} is Active and matches {dossier.get('brand_name')}.",
            evidence=evidence,
        )

    return SignalResult(
        name="nafdac",
        source="nafdac_derived",
        status="available",
        direction="neutral",
        strength=0.0,
        weight=NAFDAC_WEIGHT,
        max_weight=NAFDAC_WEIGHT,
        summary=f"NAFDAC {code_used} found with unrecognized status {raw_status!r}.",
        evidence=evidence,
        limitation=f"NAFDAC status {raw_status!r} could not be classified automatically.",
    )
