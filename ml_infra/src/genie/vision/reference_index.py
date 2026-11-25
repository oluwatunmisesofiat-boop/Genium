"""Stateless reference packaging image indexing into Object Storage and Qdrant/VectorStore."""
from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Optional
import uuid
import cv2
import numpy as np

from genie.storage.object_store import ImageStorage, LocalImageStorage
from genie.storage.vector_store import VectorStore
from genie.vision.embed import VisualEmbedder


def compute_dhash_64(img_bgr: np.ndarray) -> str:
    """Computes a 64-bit perceptual difference hash (dHash) as a 16-char hex string."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for val in diff.flatten():
        bits = (bits << 1) | int(bool(val))
    return f"{bits:016x}"


def extract_salient_regions(img_bgr: np.ndarray, max_regions: int = 5) -> Dict[str, List[int]]:
    """Auto-detects prominent high-contrast graphic/text anchor blocks on a 512x512 pack."""
    resized = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(
        cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
    )
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 1200 <= area <= 120000 and w >= 24 and h >= 16:
            candidates.append((area, [int(x), int(y), int(x + w), int(y + h)]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return {f"anchor_{idx + 1}": bbox for idx, (_, bbox) in enumerate(candidates[:max_regions])}


@dataclass
class IndexedReferencePack:
    reference_image_id: str
    storage_path: str
    sha256_hash: str
    dhash_64bit: str
    semantic_regions: Dict[str, Any]


class ReferencePackIndexer:
    """Indexes canonical genuine product pack images into Object Storage and VectorStore."""

    def __init__(
        self,
        vector_store: VectorStore,
        image_storage: Optional[ImageStorage] = None,
        embedder: Optional[VisualEmbedder] = None,
    ):
        self.vector_store = vector_store
        self.image_storage = image_storage or LocalImageStorage()
        self.embedder = embedder or VisualEmbedder()

    async def index_reference_pack(
        self,
        product_id: str,
        product_version_id: str,
        image_bgr: np.ndarray,
        brand_name: str,
        manufacturer: str,
        nafdac_number: Optional[str] = None,
        gtin_list: Optional[List[str]] = None,
        version_label: str = "v1.0",
        status: str = "active",
        valid_to: Optional[str] = None,
        shelf_life_months: int = 24,
        face_type: str = "front",
        semantic_regions: Optional[Dict[str, Any]] = None,
    ) -> IndexedReferencePack:
        """Stores a canonical reference pack image and indexes its visual vector + metadata."""
        canonical = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", canonical, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("Failed to encode reference pack image")
        img_bytes = buf.tobytes()

        sha256_hash = hashlib.sha256(img_bytes).hexdigest()
        dhash = compute_dhash_64(canonical)

        ref_id = str(uuid.uuid4())
        storage_key = f"references/{product_id}/{product_version_id}/{face_type}_{ref_id}.jpg"
        stored_path = await self.image_storage.save(storage_key, img_bytes)

        regions = semantic_regions if semantic_regions is not None else extract_salient_regions(canonical)
        embedding = self.embedder.embed_image(canonical)

        await self.vector_store.upsert(
            id=ref_id,
            vector=embedding,
            metadata={
                "reference_image_id": ref_id,
                "product_id": str(product_id),
                "product_version_id": str(product_version_id),
                "brand_name": brand_name,
                "manufacturer": manufacturer,
                "nafdac_number": nafdac_number.strip().upper() if nafdac_number else None,
                "gtin_list": [str(g).strip() for g in (gtin_list or [])],
                "version_label": version_label,
                "status": status,
                "valid_to": valid_to,
                "shelf_life_months": shelf_life_months,
                "face_type": face_type,
                "storage_path": stored_path,
                "sha256_hash": sha256_hash,
                "dhash_64bit": dhash,
                "semantic_regions": regions,
            },
        )

        return IndexedReferencePack(
            reference_image_id=ref_id,
            storage_path=stored_path,
            sha256_hash=sha256_hash,
            dhash_64bit=dhash,
            semantic_regions=regions,
        )
