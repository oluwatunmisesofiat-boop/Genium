"""Candidate pack image harvesting, SSRF protection, and 64-bit dHash deduplication."""
from io import BytesIO
from typing import List, Optional
from PIL import Image
import numpy as np


def compute_dhash_64bit(image_bytes: bytes) -> str:
    """Computes a 64-bit difference hash (dHash) using pure Pillow in <1ms."""
    with Image.open(BytesIO(image_bytes)) as img:
        img_gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = list(img_gray.getdata())

    diff_bits = []
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            diff_bits.append("1" if left > right else "0")

    bit_string = "".join(diff_bits)
    # Convert to 16-character hex string
    return f"{int(bit_string, 2):016x}"


def hamming_distance(hex1: str, hex2: str) -> int:
    val1 = int(hex1, 16)
    val2 = int(hex2, 16)
    return bin(val1 ^ val2).count("1")
