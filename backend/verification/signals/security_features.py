"""Expected physical and externally verifiable security-features signal (`independent`, weight=0.07)."""
from __future__ import annotations

from typing import Any, Dict, List

from ..schemas import ScanInput, SignalResult

SECURITY_FEATURES_WEIGHT = 0.07


def _norm_feat(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_").replace(" ", "_")


def evaluate_security_features_signal(
    scan: ScanInput,
    dossier: Dict[str, Any],
) -> SignalResult:
    """Evaluate expected security features with strict 'not observed != absent' and verifiable-only strength."""
    expected: List[Dict[str, Any]] = list(dossier.get("expected_security_features") or [])
    if not expected:
        return SignalResult(
            name="security_features",
            source="independent",
            status="not_applicable",
            direction="neutral",
            strength=0.0,
            weight=0.0,
            max_weight=0.0,
            summary="No expected security features are defined for this product.",
        )

    observed = {_norm_feat(f) for f in (scan.observed_security_features or []) if f}
    checked = {_norm_feat(f) for f in (scan.features_checked or []) if f}
    verified = {_norm_feat(f) for f in (scan.verified_features or []) if f}

    # 1. Check required features that were EXPLICITLY prompted/checked and still missing
    missing_checked: List[str] = []
    for spec in expected:
        ftype = _norm_feat(str(spec.get("feature_type") or ""))
        if spec.get("required") and ftype in checked and ftype not in observed and ftype not in verified:
            missing_checked.append(ftype)

    if missing_checked:
        return SignalResult(
            name="security_features",
            source="independent",
            status="available",
            direction="negative",
            strength=0.85,
            hard_negative="suspicious",
            weight=SECURITY_FEATURES_WEIGHT,
            max_weight=SECURITY_FEATURES_WEIGHT,
            summary=f"Required security feature(s) explicitly checked and absent: {', '.join(missing_checked)}.",
            evidence={"missing_checked": missing_checked},
        )

    # 2. Externally verified features (e.g. validated scratch PIN or manufacturer QR) unlock likely_authentic
    confirmed_verifiable: List[str] = []
    for spec in expected:
        ftype = _norm_feat(str(spec.get("feature_type") or ""))
        if spec.get("verifiable") and ftype in verified:
            confirmed_verifiable.append(ftype)

    if confirmed_verifiable:
        return SignalResult(
            name="security_features",
            source="independent",
            status="available",
            direction="positive",
            strength=0.95,
            weight=SECURITY_FEATURES_WEIGHT,
            max_weight=SECURITY_FEATURES_WEIGHT,
            unlocks_likely_authentic=True,
            summary=f"Externally verified security feature(s) confirmed: {', '.join(confirmed_verifiable)}.",
            evidence={"verified_features": confirmed_verifiable},
        )

    # 3. Visual presence alone (not externally verified) is weak positive (max 0.3, never unlocks likely_authentic)
    matched_observed: List[str] = []
    for spec in expected:
        ftype = _norm_feat(str(spec.get("feature_type") or ""))
        if ftype in observed:
            matched_observed.append(ftype)

    if matched_observed:
        return SignalResult(
            name="security_features",
            source="independent",
            status="available",
            direction="positive",
            strength=0.3,
            weight=SECURITY_FEATURES_WEIGHT,
            max_weight=SECURITY_FEATURES_WEIGHT,
            unlocks_likely_authentic=False,
            summary=(
                f"Expected feature(s) visually present ({', '.join(matched_observed)}) "
                "(weak signal — physical stickers can be imitated unless externally verified)."
            ),
            evidence={"observed_features": matched_observed},
        )

    # 4. Not observed != absent
    return SignalResult(
        name="security_features",
        source="independent",
        status="unavailable",
        direction="neutral",
        strength=0.0,
        weight=0.0,
        max_weight=SECURITY_FEATURES_WEIGHT,
        summary="Expected security features were not observed on this scan (not counted as absent).",
        limitation="Expected packaging security features were not checked or visible in this scan.",
    )
