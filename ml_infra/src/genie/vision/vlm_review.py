"""VLM Packaging Discrepancy Review and Triage using NVIDIA Nemotron multimodal reasoning.

The VLM serves strictly as a second opinion and triage descriptor to explain visual discrepancies
between query scans and canonical reference packaging. It is never the sole decider.
"""
import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import cv2
import httpx
import numpy as np

from genie.common.config import settings
from genie.common.logging import get_logger

logger = get_logger("vlm_review")


@dataclass
class VLMReviewResult:
    """Structured VLM review verdict tailored for fusion layer ingestion."""
    score: float = 0.5  # [0.0, 1.0] visual similarity score
    confidence: float = 0.5  # [0.0, 1.0] confidence in evaluation
    is_consistent: bool = True
    triage_verdict: str = "Cannot verify"  # "Consistent with genuine" | "Suspicious" | "Cannot verify"
    reasons: List[str] = field(default_factory=list)
    discrepancies: List[str] = field(default_factory=list)
    detected_nafdac: Optional[str] = None
    detected_batch: Optional[str] = None
    detected_expiry: Optional[str] = None
    raw_response: Optional[str] = None


class VLMReviewer:
    """Invokes NVIDIA Nemotron Multimodal Reasoning VLM for packaging triage and discrepancy explanation.

    Model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning (via NVIDIA NIM)
    Role: Second opinion discrepancy explainer & triage (never sole decider).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 8.0,
    ):
        self.api_key = settings.NVIDIA_API_KEY if api_key is None else api_key
        self.base_url = base_url or settings.NVIDIA_BASE_URL
        self.model = model or settings.NVIDIA_VLM_MODEL
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _encode_image_b64(img: np.ndarray, quality: int = 85) -> str:
        """Encodes an OpenCV image to base64 JPEG format."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", img, encode_param)
        if not success:
            raise ValueError("Failed to encode image to JPEG for VLM review")
        return base64.b64encode(buffer).decode("utf-8")

    def _build_payload(
        self,
        query_b64: str,
        ref_b64: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Builds an OpenAI-compatible multimodal chat completion payload."""
        prompt_text = (
            "You are an expert pharmaceutical packaging forensic inspector. "
            "Analyze the uploaded scan of pharmaceutical/consumable product packaging. "
        )

        content_items: List[Dict[str, Any]] = []

        if ref_b64:
            prompt_text += (
                "Image 1 is the QUERY SCAN taken by a consumer. "
                "Image 2 is the OFFICIAL CANONICAL REFERENCE packaging for this product.\n\n"
                "Compare the two packaging faces in detail:\n"
                "1. Brand typography, font weight, kerning, and color tone (CIELAB fidelity).\n"
                "2. Regulatory symbols: NAFDAC registration badge/number format and placement.\n"
                "3. Inkjet batch number and expiry date printing style (dot-matrix vs laser).\n"
                "4. Identify any localized tampering, spelling errors, or missing security seals.\n\n"
            )
            content_items.append({"type": "text", "text": prompt_text})
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"},
            })
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"},
            })
        else:
            prompt_text += (
                "Inspect this query packaging scan. Read all visible printed text, "
                "especially the NAFDAC registration number, Batch/Lot number, and Expiry date. "
                "Identify brand name, manufacturer, and evaluate print sharpness, tampering signs "
                "(e.g. torn labels, hand-drawn pen markings, missing regulatory seals), and layout consistency. "
                "Note: If two boxes are shown side-by-side representing an outer multipack carton (e.g. '10 x 1 x 6') "
                "and its inner unit pack (e.g. '1 x 6'), that outer-vs-inner pack count difference is normal and NOT a defect.\n\n"
            )
            content_items.append({"type": "text", "text": prompt_text})
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"},
            })

        format_instruction = (
            "Respond ONLY with a valid JSON object strictly matching this schema, with no markdown code blocks:\n"
            "{\n"
            '  "score": <float between 0.0 and 1.0 representing packaging authenticity/consistency>,\n'
            '  "confidence": <float between 0.0 and 1.0 representing your confidence in inspection>,\n'
            '  "is_consistent": <true if consistent with genuine packaging, false if suspicious>,\n'
            '  "triage_verdict": <"Consistent with genuine" | "Suspicious" | "Cannot verify">,\n'
            '  "detected_nafdac": <string or null, e.g. "A4-1234">,\n'
            '  "detected_batch": <string or null, e.g. "B89412A">,\n'
            '  "detected_expiry": <string or null, e.g. "12/2027">,\n'
            '  "reasons": [<concise list of 1-3 supporting observations>],\n'
            '  "discrepancies": [<list of specific packaging defects or mismatches, or empty array>]\n'
            "}"
        )
        content_items.append({"type": "text", "text": format_instruction})

        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a forensic packaging auditor. Output strict JSON only.",
                },
                {"role": "user", "content": content_items},
            ],
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 0.95,
        }

    def _parse_response(self, raw_text: Optional[str]) -> VLMReviewResult:
        """Parses model JSON output, stripping any markdown wrappers."""
        if not raw_text:
            return VLMReviewResult(
                score=0.5,
                confidence=0.2,
                is_consistent=True,
                triage_verdict="Cannot verify",
                reasons=["VLM returned empty response content."],
            )
        clean_text = raw_text.strip()
        # Remove markdown code fences if present
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text)

        try:
            data = json.loads(clean_text)
            return VLMReviewResult(
                score=float(data.get("score", 0.5)),
                confidence=float(data.get("confidence", 0.5)),
                is_consistent=bool(data.get("is_consistent", True)),
                triage_verdict=str(data.get("triage_verdict", "Cannot verify")),
                reasons=list(data.get("reasons", [])),
                discrepancies=list(data.get("discrepancies", [])),
                detected_nafdac=data.get("detected_nafdac"),
                detected_batch=data.get("detected_batch"),
                detected_expiry=data.get("detected_expiry"),
                raw_response=raw_text,
            )
        except Exception as e:
            logger.warning(f"Failed to parse VLM JSON output: {e}. Output was: {raw_text[:200]}")
            # Fallback extraction via regex
            score_match = re.search(r'"score"\s*:\s*([0-9.]+)', raw_text)
            score = float(score_match.group(1)) if score_match else 0.5
            return VLMReviewResult(
                score=score,
                confidence=0.4,
                is_consistent=score >= 0.70,
                triage_verdict="Cannot verify",
                reasons=["VLM review completed with partial formatting"],
                raw_response=raw_text,
            )

    async def review_packaging_async(
        self,
        query_image: np.ndarray,
        reference_image: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> VLMReviewResult:
        """Asynchronously queries the NVIDIA Nemotron VLM for discrepancy explanation."""
        if not self.api_key:
            logger.info("NVIDIA_API_KEY not configured. Skipping VLM second-opinion review.")
            return VLMReviewResult(
                score=0.5,
                confidence=0.1,
                is_consistent=True,
                triage_verdict="Cannot verify",
                reasons=["VLM review skipped: NVIDIA API key not configured."],
            )

        try:
            query_b64 = self._encode_image_b64(query_image)
            ref_b64 = self._encode_image_b64(reference_image) if reference_image is not None else None
            payload = self._build_payload(query_b64, ref_b64, context)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            invoke_url = f"{self.base_url.rstrip('/')}/chat/completions"

            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(invoke_url, headers=headers, json=payload)
                resp.raise_for_status()
                res_json = resp.json()

            msg = res_json["choices"][0]["message"]
            raw_content = msg.get("content") or msg.get("reasoning_content") or ""
            return self._parse_response(raw_content)

        except Exception as e:
            logger.warning(f"VLM review request encountered error: {e}. Falling back gracefully.")
            return VLMReviewResult(
                score=0.5,
                confidence=0.1,
                is_consistent=True,
                triage_verdict="Cannot verify",
                reasons=[f"VLM second opinion temporarily unavailable ({type(e).__name__})"],
            )

    def review_packaging(
        self,
        query_image: np.ndarray,
        reference_image: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> VLMReviewResult:
        """Synchronous wrapper for VLM discrepancy review."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Run in thread executor if inside existing running loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(
                        asyncio.run,
                        self.review_packaging_async(query_image, reference_image, context)
                    ).result()
            return loop.run_until_complete(
                self.review_packaging_async(query_image, reference_image, context)
            )
        except Exception:
            return asyncio.run(
                self.review_packaging_async(query_image, reference_image, context)
            )
