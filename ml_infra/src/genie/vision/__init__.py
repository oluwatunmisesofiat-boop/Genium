"""Computer Vision subsystem: quality gating, pack rectification, embeddings, alignment, and anomalies."""
from genie.vision.anomaly import PatchCoreAnomalyDetector
from genie.vision.embed import VisualEmbedder
from genie.vision.match import GeometricMatcher
from genie.vision.ocr import PackagingOCR
from genie.vision.pack_detect import PackDetectorRectifier
from genie.vision.print_forensics import PrintForensicsAnalyzer
from genie.vision.quality import ImageQualityGate, QualityGateResult

__all__ = [
    "ImageQualityGate",
    "QualityGateResult",
    "PackDetectorRectifier",
    "VisualEmbedder",
    "GeometricMatcher",
    "PackagingOCR",
    "PatchCoreAnomalyDetector",
    "PrintForensicsAnalyzer",
]
