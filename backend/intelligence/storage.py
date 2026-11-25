"""Swappable reference image storage (local disk implementation for v1)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_SAFE_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def default_image_dir() -> Path:
    env_dir = os.environ.get("GENIE_IMAGE_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parent.parent / "data" / "images"


def _slugify_key(product_key: str) -> str:
    cleaned = _SAFE_SLUG_RE.sub("_", (product_key or "").strip().lower()).strip("_")
    return cleaned or "unknown"


@runtime_checkable
class ImageStorage(Protocol):
    def save(self, product_key: str, sha256: str, data: bytes, mime_type: str) -> str: ...
    def exists(self, storage_path: str) -> bool: ...
    def read(self, storage_path: str) -> bytes: ...


class LocalImageStorage:
    """Stores verified packaging images under `<root_dir>/<product_key>/<sha256[:16]><ext>`."""

    def __init__(self, root_dir: Optional[Path | str] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else default_image_dir()

    def resolve(self, storage_path: str) -> Path:
        root_resolved = self.root_dir.resolve()
        candidate = (root_resolved / storage_path).resolve()
        if candidate != root_resolved and root_resolved not in candidate.parents:
            raise ValueError(f"Invalid storage path outside root: {storage_path!r}")
        return candidate

    def save(self, product_key: str, sha256: str, data: bytes, mime_type: str) -> str:
        slug = _slugify_key(product_key)
        clean_mime = (mime_type or "").split(";", 1)[0].strip().lower()
        ext = _EXT_BY_MIME.get(clean_mime, ".jpg")
        rel_path = f"{slug}/{sha256[:16]}{ext}"
        target = self.resolve(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        return rel_path

    def exists(self, storage_path: str) -> bool:
        try:
            return self.resolve(storage_path).is_file()
        except ValueError:
            return False

    def read(self, storage_path: str) -> bytes:
        return self.resolve(storage_path).read_bytes()
