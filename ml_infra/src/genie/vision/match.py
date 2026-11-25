"""Geometric alignment and feature matching via OpenCV ORB keypoints and RANSAC homography (no PyTorch)."""
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class AlignmentResult:
    matched: bool
    inlier_ratio: float
    num_matches: int
    num_inliers: int
    reprojection_error: float
    homography_matrix: Optional[np.ndarray] = None
    aligned_image: Optional[np.ndarray] = None


class GeometricMatcher:
    """Matches query pack features against canonical reference faces.

    Enforces minimum inlier counts, inlier ratios, and low reprojection error.
    """

    def __init__(
        self,
        min_inliers: int = 25,
        min_inlier_ratio: float = 0.35,
        max_reprojection_error: float = 4.0,
    ):
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reprojection_error = max_reprojection_error

    def match(
        self,
        query_img: np.ndarray,
        reference_img: np.ndarray,
        output_size: Tuple[int, int] = (512, 512),
    ) -> AlignmentResult:
        """Aligns query image against reference pack using keypoint homography."""
        # Ensure grayscale representations for feature detection
        q_gray = (
            cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
            if len(query_img.shape) == 3
            else query_img
        )
        r_gray = (
            cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
            if len(reference_img.shape) == 3
            else reference_img
        )

        # High-density ORB feature detector calibrated for packaging text and micro-graphics
        detector = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=10,
        )

        kp1, des1 = detector.detectAndCompute(q_gray, None)
        kp2, des2 = detector.detectAndCompute(r_gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return AlignmentResult(
                matched=False,
                inlier_ratio=0.0,
                num_matches=0,
                num_inliers=0,
                reprojection_error=999.0,
            )

        # BFMatcher with Hamming distance and k-NN ratio test (Lowe's test)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn_matches = matcher.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test (0.75 threshold)
        good_matches = []
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.78 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            return AlignmentResult(
                matched=False,
                inlier_ratio=0.0,
                num_matches=len(good_matches),
                num_inliers=0,
                reprojection_error=999.0,
            )

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Robust RANSAC homography estimation
        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.max_reprojection_error
        )

        if H is None or mask is None:
            return AlignmentResult(
                matched=False,
                inlier_ratio=0.0,
                num_matches=len(good_matches),
                num_inliers=0,
                reprojection_error=999.0,
            )

        inlier_mask = mask.ravel() == 1
        num_inliers = int(np.sum(inlier_mask))
        inlier_ratio = float(num_inliers / len(good_matches)) if good_matches else 0.0

        # Calculate mean reprojection error across inliers
        reproj_error = 999.0
        if num_inliers > 0:
            inlier_src = src_pts[inlier_mask]
            inlier_dst = dst_pts[inlier_mask]

            transformed_src = cv2.perspectiveTransform(inlier_src, H)
            errors = np.linalg.norm(transformed_src - inlier_dst, axis=2)
            reproj_error = float(np.mean(errors))

        # Check alignment criteria
        is_matched = (
            num_inliers >= self.min_inliers
            and inlier_ratio >= self.min_inlier_ratio
            and reproj_error <= self.max_reprojection_error
        )

        aligned_img = None
        if is_matched:
            aligned_img = cv2.warpPerspective(
                query_img, H, output_size, flags=cv2.INTER_LINEAR
            )

        return AlignmentResult(
            matched=is_matched,
            inlier_ratio=inlier_ratio,
            num_matches=len(good_matches),
            num_inliers=num_inliers,
            reprojection_error=reproj_error,
            homography_matrix=H,
            aligned_image=aligned_img,
        )
