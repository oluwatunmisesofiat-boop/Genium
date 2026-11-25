"""Unit tests for Visual Feature Embedding and Vector Retrieval."""
import cv2
import numpy as np
import pytest

from genie.storage.vector_store import InMemoryVectorStore
from genie.vision.embed import VisualEmbedder


def test_visual_embedder_unit_vector():
    embedder = VisualEmbedder()
    img = np.ones((512, 512, 3), dtype=np.uint8) * 128
    vec = embedder.embed_image(img)

    assert len(vec) == 768
    # Verify L2 norm is 1.0 (within float precision)
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, 1e-4) == 1.0


@pytest.mark.asyncio
async def test_vector_store_upsert_and_search():
    store = InMemoryVectorStore()
    embedder = VisualEmbedder()

    # Product A: Green packaging
    img_a = np.zeros((512, 512, 3), dtype=np.uint8)
    img_a[:, :] = [20, 180, 40]
    vec_a = embedder.embed_image(img_a)

    # Product B: Red packaging
    img_b = np.zeros((512, 512, 3), dtype=np.uint8)
    img_b[:, :] = [30, 20, 220]
    vec_b = embedder.embed_image(img_b)

    await store.upsert("prod_coartem", vec_a, {
        "product_id": "coartem_uuid",
        "gtin_list": ["06151234567891"],
        "nafdac_number": "A4-6238",
    })
    await store.upsert("prod_panadol", vec_b, {
        "product_id": "panadol_uuid",
        "gtin_list": ["06159999999999"],
        "nafdac_number": "04-9999",
    })

    assert await store.count() == 2

    # Query with a slightly varied version of Product A
    query_img = img_a.copy()
    cv2.circle(query_img, (256, 256), 30, (255, 255, 255), -1)
    query_vec = embedder.embed_image(query_img)

    # 1. Unfiltered Top-1 Search
    results = await store.search(query_vec, top_k=2)
    assert len(results) == 2
    assert results[0].id == "prod_coartem"
    assert results[0].score > results[1].score

    # 2. Metadata Filtered Search by GTIN
    filtered = await store.search(query_vec, top_k=5, filter_metadata={"gtin_list": "06151234567891"})
    assert len(filtered) == 1
    assert filtered[0].id == "prod_coartem"
