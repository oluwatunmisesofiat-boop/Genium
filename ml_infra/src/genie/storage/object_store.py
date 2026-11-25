"""Object store adapter protocol and local filesystem implementation."""
import os
from pathlib import Path
from typing import Optional, Protocol


class ImageStorage(Protocol):
    async def save(self, key: str, data: bytes) -> str:
        """Saves image bytes and returns the stored path or URI."""
        ...

    async def read(self, key: str) -> bytes:
        """Reads image bytes for the given storage key."""
        ...

    async def exists(self, key: str) -> bool:
        """Checks if the key exists."""
        ...


class LocalImageStorage:
    def __init__(self, base_dir: Path | str = "./data/images"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, key: str, data: bytes) -> str:
        target_path = self.base_dir / key
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(data)
        return str(target_path)

    async def read(self, key: str) -> bytes:
        target_path = self.base_dir / key
        with open(target_path, "rb") as f:
            return f.read()

    async def exists(self, key: str) -> bool:
        return (self.base_dir / key).is_file()
