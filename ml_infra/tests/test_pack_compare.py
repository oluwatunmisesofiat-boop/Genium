"""Tests for reference-to-query packaging comparison (color, element positioning, layout, and stateless pipeline)."""
import base64
import uuid
import cv2
import numpy as np
import pytest

from genie.schemas.scan import ScanImage, ScanRequest
from genie.storage.object_store import LocalImageStorage
from genie.storage.vector_store import InMemoryVectorStore
from genie.verification.pipeline import VerificationPipeline
from genie.vision.pack_compare import PackComparator
from genie.vision.reference_index import ReferencePackIndexer


def _make_synthetic_pack() -> tuple[np.ndarray, dict]:
    """Creates a textured, high-contrast synthetic 512x512 pharmaceutical box with distinct landmarks."""
    img = np.full((512, 512, 3), (215, 218, 222), dtype=np.uint8)

    # Background geometric grid/texture for reliable ORB keypoints
    rng = np.random.default_rng(42)
    for _ in range(80):
        x, y = int(rng.integers(20, 490)), int(rng.integers(20, 490))
        r = int(rng.integers(3, 10))
        cv2.circle(img, (x, y), r, (185, 190, 198), -1)

    # Brand header banner (Blue)
    cv2.rectangle(img, (40, 40), (472, 135), (170, 85, 25), -1)
    cv2.putText(img, "GENIEMED 80/480", (65, 102), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    # NAFDAC badge (Green box)
    cv2.rectangle(img, (50, 175), (210, 255), (45, 145, 45), -1)
    cv2.putText(img, "NAFDAC B4-6458", (60, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    # Active ingredient block
    cv2.rectangle(img, (240, 175), (465, 275), (225, 230, 235), -1)
    cv2.rectangle(img, (240, 175), (465, 275), (80, 80, 80), 2)
    cv2.putText(img, "Artemether 80mg", (255, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(img, "Lumefantrine 480mg", (255, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2)

    # Simulated 1D barcode block at bottom
    cv2.rectangle(img, (60, 355), (300, 455), (235, 235, 235), -1)
    for bx in range(72, 290, 8):
        thick = 3 if (bx // 8) % 2 == 0 else 2
        cv2.line(img, (bx, 365), (bx, 445), (10, 10, 10), thick)

    # Security seal on bottom-right
    cv2.circle(img, (410, 405), 42, (35, 165, 215), -1)
    cv2.putText(img, "SEAL", (386, 412), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    regions = {
        "brand_logo": [40, 40, 472, 135],
        "nafdac_badge": [50, 175, 210, 255],
        "barcode": [60, 355, 300, 455],
        "security_seal": [365, 360, 455, 450],
    }
    return img, regions


def test_genuine_pack_with_lighting_variation_passes():
    ref, regions = _make_synthetic_pack()
    # Simulate slightly warmer/brighter smartphone photo
    query = np.clip(ref.astype(np.float32) * 1.08 + 8.0, 0, 255).astype(np.uint8)

    comparator = PackComparator()
    res = comparator.compare(query, ref, semantic_regions=regions)

    assert res.alignment_matched is True
    assert res.color_score >= 0.80
    assert res.layout_score >= 0.80
    assert res.overall_similarity_score >= 0.75
    assert res.is_counterfeit_suspect is False
    assert all(c.passed for c in res.element_checks)


def test_color_shifted_counterfeit_is_flagged():
    ref, regions = _make_synthetic_pack()
    # Swap B and R channels so blue/green packaging becomes orange/red counterfeit ink
    query = ref[:, :, ::-1].copy()

    comparator = PackComparator()
    res = comparator.compare(query, ref, semantic_regions=regions)

    assert res.delta_e_ab > 15.0
    assert res.color_score < 0.55
    assert res.is_counterfeit_suspect is True
    assert any("color mismatch" in msg.lower() for msg in res.inconsistencies)


def test_shifted_and_missing_elements_are_flagged():
    ref, regions = _make_synthetic_pack()
    query = ref.copy()

    # Erase NAFDAC badge from original spot and shift it down-right by 32px
    badge = ref[175:255, 50:210].copy()
    query[175:255, 50:210] = (215, 218, 222)
    query[207:287, 82:242] = badge

    # Remove security seal completely
    cv2.circle(query, (410, 405), 45, (215, 218, 222), -1)

    comparator = PackComparator()
    res = comparator.compare(query, ref, semantic_regions=regions)

    checks_by_name = {c.name: c for c in res.element_checks}
    assert checks_by_name["nafdac_badge"].status == "shifted"
    assert checks_by_name["nafdac_badge"].shift_px > 20.0
    assert checks_by_name["security_seal"].status == "missing_or_altered"
    assert res.is_counterfeit_suspect is True
    assert any("nafdac_badge" in msg for msg in res.inconsistencies)
    assert any("security_seal" in msg for msg in res.inconsistencies)


@pytest.mark.asyncio
async def test_reference_indexer_and_pipeline_comparison(tmp_path):
    vector_store = InMemoryVectorStore()
    storage = LocalImageStorage(base_dir=tmp_path / "images")

    ref_img, regions = _make_synthetic_pack()

    indexer = ReferencePackIndexer(vector_store=vector_store, image_storage=storage)
    ref_record = await indexer.index_reference_pack(
        product_id=str(uuid.uuid4()),
        product_version_id=str(uuid.uuid4()),
        image_bgr=ref_img,
        brand_name="GENIEMED",
        manufacturer="GENIE PHARMA",
        nafdac_number="B4-6458",
        gtin_list=["06150000000012"],
        version_label="v1.0",
        face_type="front",
        semantic_regions=regions,
    )
    assert ref_record.sha256_hash
    assert len(ref_record.dhash_64bit) == 16

    pipeline = VerificationPipeline(vector_store=vector_store)

    # 1. Genuine scan
    _, ok_buf = cv2.imencode(".jpg", ref_img)
    ok_b64 = base64.b64encode(ok_buf.tobytes()).decode("utf-8")
    res_ok = await pipeline.verify_scan(
        ScanRequest(
            images=[ScanImage(image_bytes_base64=ok_b64, face_hint="front")],
            manual_nafdac="B4-6458",
        )
    )
    assert res_ok.signals.visual_similarity_score >= 0.80

    # 2. Counterfeit color-altered scan
    fake_img = ref_img[:, :, ::-1].copy()
    _, fake_buf = cv2.imencode(".jpg", fake_img)
    fake_b64 = base64.b64encode(fake_buf.tobytes()).decode("utf-8")
    res_fake = await pipeline.verify_scan(
        ScanRequest(
            images=[ScanImage(image_bytes_base64=fake_b64, face_hint="front")],
            manual_nafdac="B4-6458",
        )
    )
    assert res_fake.signals.visual_similarity_score < res_ok.signals.visual_similarity_score
    assert any("color mismatch" in r.lower() for r in res_fake.signals.reasons)
