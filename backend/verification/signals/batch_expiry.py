"""Batch manufacturing/expiry date and SmPC shelf-life plausibility signal (`nafdac_derived`, weight=0.10)."""
from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from ..schemas import ScanInput, SignalResult

BATCH_EXPIRY_WEIGHT = 0.10
SHELF_LIFE_TOLERANCE_MONTHS = 2

_MONTH_NAMES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def parse_pack_date(raw: Optional[str], *, end_of_month: bool = False) -> Optional[date]:
    """Parse common pharmaceutical pack date formats (YYYY-MM-DD, YYYY-MM, MM/YYYY, Mon YYYY)."""
    text = (raw or "").strip()
    if not text:
        return None

    # ISO YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    # YYYY-MM or YYYY/MM
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})", text)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            day = calendar.monthrange(y, mo)[1] if end_of_month else 1
            return date(y, mo, day)

    # DD/MM/YYYY or DD-MM-YYYY
    m = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", text)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            return None

    # MM/YYYY or MM-YYYY
    m = re.fullmatch(r"(\d{1,2})[-/](\d{4})", text)
    if m:
        mo, y = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            day = calendar.monthrange(y, mo)[1] if end_of_month else 1
            return date(y, mo, day)

    # MM/YY or MM-YY (common on medicine blisters, e.g. "06/27")
    m = re.fullmatch(r"(\d{1,2})[-/](\d{2})", text)
    if m:
        mo, yy = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            y = 2000 + yy
            day = calendar.monthrange(y, mo)[1] if end_of_month else 1
            return date(y, mo, day)

    # Mon YYYY (e.g. "Apr 2024", "April 2024")
    m = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{4})", text)
    if m:
        mo = _MONTH_NAMES.get(m.group(1).lower())
        y = int(m.group(2))
        if mo:
            day = calendar.monthrange(y, mo)[1] if end_of_month else 1
            return date(y, mo, day)

    return None


def evaluate_batch_expiry_signal(
    scan: ScanInput,
    dossier: Dict[str, Any],
) -> SignalResult:
    """Validate expiry status, MFG < EXP ordering, and MFG-to-EXP gap against SmPC shelf_life_months."""
    mfg = parse_pack_date(scan.mfg_date, end_of_month=False)
    exp = parse_pack_date(scan.expiry_date, end_of_month=True)
    today = parse_pack_date(scan.scan_date, end_of_month=False) or datetime.now(timezone.utc).date()

    if mfg is None and exp is None:
        return SignalResult(
            name="batch_expiry",
            source="nafdac_derived",
            status="unavailable",
            direction="neutral",
            strength=0.0,
            weight=0.0,
            max_weight=BATCH_EXPIRY_WEIGHT,
            summary="No manufacturing or expiry date was provided.",
            limitation="Manufacturing and expiry dates were not scanned.",
        )

    evidence: Dict[str, Any] = {
        "mfg_date": mfg.isoformat() if mfg else None,
        "expiry_date": exp.isoformat() if exp else None,
        "scan_date": today.isoformat(),
        "smpc_shelf_life_months": dossier.get("shelf_life_months"),
    }

    if mfg is not None and exp is not None and mfg >= exp:
        return SignalResult(
            name="batch_expiry",
            source="nafdac_derived",
            status="available",
            direction="negative",
            strength=1.0,
            hard_negative="suspicious",
            weight=BATCH_EXPIRY_WEIGHT,
            max_weight=BATCH_EXPIRY_WEIGHT,
            summary=f"Impossible dates: manufacturing date ({mfg}) is on or after expiry date ({exp}).",
            evidence=evidence,
        )

    if exp is not None and exp < today:
        return SignalResult(
            name="batch_expiry",
            source="nafdac_derived",
            status="available",
            direction="negative",
            strength=0.9,
            hard_negative="suspicious",
            weight=BATCH_EXPIRY_WEIGHT,
            max_weight=BATCH_EXPIRY_WEIGHT,
            summary=f"Product pack is expired (expiry date {exp} is before {today}).",
            evidence=evidence,
        )

    shelf_life = dossier.get("shelf_life_months")
    if mfg is not None and exp is not None and isinstance(shelf_life, int) and shelf_life > 0:
        span_months = (exp.year - mfg.year) * 12 + (exp.month - mfg.month)
        evidence["observed_shelf_life_months"] = span_months
        if abs(span_months - shelf_life) > SHELF_LIFE_TOLERANCE_MONTHS:
            return SignalResult(
                name="batch_expiry",
                source="nafdac_derived",
                status="available",
                direction="negative",
                strength=0.9,
                hard_negative="suspicious",
                weight=BATCH_EXPIRY_WEIGHT,
                max_weight=BATCH_EXPIRY_WEIGHT,
                summary=(
                    f"MFG-to-EXP span is {span_months} months, contradicting SmPC shelf life "
                    f"of {shelf_life} months (±{SHELF_LIFE_TOLERANCE_MONTHS} mo tolerance)."
                ),
                evidence=evidence,
            )
        return SignalResult(
            name="batch_expiry",
            source="nafdac_derived",
            status="available",
            direction="positive",
            strength=0.95,
            weight=BATCH_EXPIRY_WEIGHT,
            max_weight=BATCH_EXPIRY_WEIGHT,
            summary=f"Pack is unexpired and {span_months}-month shelf life matches SmPC ({shelf_life} months).",
            evidence=evidence,
        )

    return SignalResult(
        name="batch_expiry",
        source="nafdac_derived",
        status="available",
        direction="positive",
        strength=0.7,
        weight=BATCH_EXPIRY_WEIGHT,
        max_weight=BATCH_EXPIRY_WEIGHT,
        summary="Pack expiry date is valid and unexpired.",
        evidence=evidence,
        limitation=(
            "SmPC shelf life was unavailable; only basic expiry sanity was checked."
            if not shelf_life
            else None
        ),
    )
