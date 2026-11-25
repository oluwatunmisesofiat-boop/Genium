"""Run the full Genie verification pipeline on all images in /ml_infra/images."""
import asyncio
import base64
import json
from pathlib import Path
import cv2

from genie.barcode.decode import MultiDecoder
from genie.common.exceptions import QualityGateError
from genie.schemas.scan import ScanImage, ScanRequest
from genie.storage.vector_store import InMemoryVectorStore
from genie.verification.pipeline import VerificationPipeline
from genie.vision.pack_compare import PackComparator


async def main():
    images_dir = Path("images")
    image_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    if not image_files:
        print("No images found in images/")
        return

    vector_store = InMemoryVectorStore()
    results = []

    if True:
        pipeline = VerificationPipeline(
            vector_store=vector_store,
            enable_vlm_review=True,
        )
        # Run VLM only once per image inside verify_scan + standalone VLM forensic review
        pipeline.ocr.enable_vlm = False
        pipeline.vlm_reviewer.timeout_seconds = 60.0

        for img_path in image_files:
            print(f"\n=== Processing: {img_path.name} ===", flush=True)
            img_bytes = img_path.read_bytes()
            img_np = cv2.imdecode(
                __import__("numpy").frombuffer(img_bytes, __import__("numpy").uint8),
                cv2.IMREAD_COLOR,
            )

            # 1. Quality Gate check
            q_res = pipeline.quality_gate.evaluate(img_np)
            print("  [1] Quality Gate done", flush=True)
            # 2. Barcode decode check
            rect_res = pipeline.pack_rectifier.rectify(img_np)
            barcode_res = MultiDecoder.decode(rect_res.rectified_image) or MultiDecoder.decode(img_np)
            print("  [2] Barcode decode done", flush=True)
            # 3. OCR check (local)
            ocr_res = pipeline.ocr.extract_text(img_np)
            print("  [3] OCR done", flush=True)
            # 4. VLM forensic review (standalone)
            vlm_res = await pipeline.vlm_reviewer.review_packaging_async(img_np)
            print("  [4] VLM review done", flush=True)

            # 5. Full VerificationPipeline execution (with VLM second-opinion triage enabled)
            pipeline.enable_vlm_review = True
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            scan_req = ScanRequest(
                images=[ScanImage(image_bytes_base64=img_b64, face_hint="front")],
                manual_nafdac=ocr_res.nafdac_number or vlm_res.detected_nafdac,
            )

            pipeline_out = {}
            try:
                resp = await pipeline.verify_scan(scan_req)
                pipeline_out = {
                    "verdict": resp.verdict.value,
                    "confidence": round(resp.confidence, 4),
                    "matched_product_name": resp.matched_product_name,
                    "explanation": resp.explanation,
                    "signals": resp.signals.model_dump(),
                    "execution_time_ms": round(resp.execution_time_ms, 2),
                }
            except QualityGateError as qe:
                pipeline_out = {
                    "verdict": "REJECTED_BY_QUALITY_GATE",
                    "error": str(qe),
                    "details": qe.details,
                }

            entry = {
                "filename": img_path.name,
                "resolution": f"{img_np.shape[1]}x{img_np.shape[0]}",
                "quality_gate": {
                    "passed": q_res.passed,
                    "blur_score": round(q_res.blur_score, 2),
                    "glare_ratio": round(q_res.glare_ratio, 4),
                    "guidance_codes": q_res.guidance_codes,
                },
                "barcode": barcode_res,
                "ocr": {
                    "nafdac_number": ocr_res.nafdac_number,
                    "batch_number": ocr_res.batch_number,
                    "expiry_date": ocr_res.expiry_date,
                    "raw_text_lines": ocr_res.raw_text_lines[:15],
                },
                "vlm_forensic_review": {
                    "score": vlm_res.score,
                    "confidence": vlm_res.confidence,
                    "is_consistent": vlm_res.is_consistent,
                    "triage_verdict": vlm_res.triage_verdict,
                    "detected_nafdac": vlm_res.detected_nafdac,
                    "detected_batch": vlm_res.detected_batch,
                    "detected_expiry": vlm_res.detected_expiry,
                    "reasons": vlm_res.reasons,
                    "discrepancies": vlm_res.discrepancies,
                },
                "pipeline_result": pipeline_out,
            }
            results.append(entry)
            print(
                f"  Pipeline Verdict: {pipeline_out.get('verdict')} | "
                f"VLM Triage: {vlm_res.triage_verdict} (score={vlm_res.score}) | "
                f"NAFDAC: {ocr_res.nafdac_number or vlm_res.detected_nafdac}"
            )

        # Pairwise visual comparison if >= 2 images exist
        pairwise = []
        if len(image_files) >= 2:
            comparator = PackComparator()
            imgs = {p.name: cv2.imread(str(p), cv2.IMREAD_COLOR) for p in image_files}
            names = list(imgs.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    comp = comparator.compare(imgs[names[i]], imgs[names[j]])
                    pairwise.append({
                        "image_a": names[i],
                        "image_b": names[j],
                        "overall_similarity_score": round(comp.overall_similarity_score, 4),
                        "is_counterfeit_suspect": comp.is_counterfeit_suspect,
                        "alignment_matched": comp.alignment_matched,
                        "inlier_ratio": round(comp.inlier_ratio, 4),
                        "color_score": round(comp.color_score, 4),
                        "delta_e_ab": round(comp.delta_e_ab, 2),
                        "layout_score": round(comp.layout_score, 4),
                        "structural_ssim": round(comp.structural_ssim, 4),
                        "inconsistencies": comp.inconsistencies,
                    })

    report = {"images": results, "pairwise_comparisons": pairwise}
    out_path = images_dir / "pipeline_verdicts.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved full pipeline report to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
