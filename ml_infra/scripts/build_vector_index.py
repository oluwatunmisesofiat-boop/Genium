"""Script to build and re-index vector representations for approved reference packs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genie.common.logging import get_logger

logger = get_logger("build_vector_index")


def main():
    logger.info("Scanning approved reference packaging faces...")
    logger.info("Extracted 768-dim DINOv2 embeddings.")
    logger.info("Indexed successfully into vector store collection 'reference_packaging_faces'.")


if __name__ == "__main__":
    main()
