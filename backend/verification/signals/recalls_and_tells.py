"""Recall alerts and known-counterfeit-tells safety gate (weight=0.00; hard negative on hit)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..schemas import CounterfeitTellHit, ScanInput, SignalResult


def _norm_batch(val: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (val or "").strip()).upper()


def evaluate_recalls_and_tells_signal(
    scan: ScanInput,
    dossier: Dict[str, Any],
) -> Tuple[SignalResult, List[CounterfeitTellHit]]:
    """Check submitted batch and OCR text against known_counterfeit_tells and recall_alerts.

    This is a negative-only safety gate (weight=0.0 in positive score so clear checks
    never dilute genuine scores; any hit fires `likely_counterfeit` at any coverage).
    """
    hits: List[CounterfeitTellHit] = []
    submitted_batch_raw = (scan.batch_number or "").strip()
    submitted_batch = _norm_batch(submitted_batch_raw)
    ocr_low = (scan.ocr_text or "").lower()

    # 1. Structured known_counterfeit_tells
    for tell in dossier.get("known_counterfeit_tells") or []:
        t_type = str(tell.get("tell_type") or "other")
        summary = str(tell.get("summary") or "Known counterfeit tell matched")
        source_kind = (
            "nafdac_derived" if tell.get("source_kind") == "nafdac_derived" else "independent"
        )
        affected = [str(b) for b in (tell.get("affected_batches") or []) if str(b).strip()]

        if submitted_batch and any(_norm_batch(b) == submitted_batch for b in affected):
            hits.append(
                CounterfeitTellHit(
                    tell_type=t_type,
                    summary=summary,
                    matched_on=f"batch_number={submitted_batch_raw}",
                    alert_ref=tell.get("alert_ref"),
                    source_url=tell.get("source_url"),
                    source_kind=source_kind,
                )
            )
            continue

        fake_ind = str(tell.get("fake_indicator") or "").strip().lower()
        if fake_ind and len(fake_ind) >= 4 and ocr_low and fake_ind in ocr_low:
            hits.append(
                CounterfeitTellHit(
                    tell_type=t_type,
                    summary=summary,
                    matched_on=f"ocr_indicator={tell.get('fake_indicator')}",
                    alert_ref=tell.get("alert_ref"),
                    source_url=tell.get("source_url"),
                    source_kind=source_kind,
                )
            )

    # 2. Free-text recall_alerts fallback for batch numbers (e.g. "Batch No. 562626")
    if submitted_batch_raw and not any("batch_number=" in h.matched_on for h in hits):
        batch_pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(submitted_batch_raw)}(?![A-Za-z0-9])", re.IGNORECASE)
        for alert in dossier.get("recall_alerts") or []:
            alert_text = str(alert or "")
            if batch_pat.search(alert_text):
                hits.append(
                    CounterfeitTellHit(
                        tell_type="compromised_batch",
                        summary=alert_text,
                        matched_on=f"recall_alert_batch={submitted_batch_raw}",
                        source_kind="nafdac_derived" if "nafdac" in alert_text.lower() else "independent",
                    )
                )

    if hits:
        primary_source = (
            "independent"
            if any(h.source_kind == "independent" for h in hits)
            else "nafdac_derived"
        )
        return (
            SignalResult(
                name="recalls_and_tells",
                source=primary_source,
                status="available",
                direction="negative",
                strength=1.0,
                hard_negative="likely_counterfeit",
                weight=0.0,
                max_weight=0.0,
                summary=f"Matched {len(hits)} known recall/counterfeit tell(s): {hits[0].summary}",
                evidence={"hits": [h.model_dump() for h in hits]},
            ),
            hits,
        )

    recall_count = len(dossier.get("recall_alerts") or []) + len(
        dossier.get("known_counterfeit_tells") or []
    )
    note = (
        f"No match across {recall_count} tracked recall/counterfeit alert(s) (clear check is neutral, not proof)."
        if recall_count
        else "No active recall alerts or known counterfeit tells matched (clear check is neutral, not proof)."
    )
    return (
        SignalResult(
            name="recalls_and_tells",
            source="independent",
            status="available",
            direction="neutral",
            strength=0.0,
            weight=0.0,
            max_weight=0.0,
            summary=note,
            evidence={"tracked_alerts_checked": recall_count},
        ),
        [],
    )
