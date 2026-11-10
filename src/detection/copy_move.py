"""
Copy-Move (Cloning) Detection via Keypoint Matching & RANSAC Verification.

Theoretical Foundation:
1. Feature Extraction: SIFT (Scale-Invariant Feature Transform) / ORB descriptors.
2. Variance Gating: Exclude keypoints in uniform textureless zones (sigma^2(x, y) < tau_flat).
3. Generalized 2NN (g2NN) Matching:
   Ratio test dist(d_1) / dist(d_2) < rho, enforcing minimum spatial displacement
   ||p_1 - p_2||_2 > d_min to eliminate false auto-correlations.
4. RANSAC Affine Verification:
   Fits affine model p' = H * p to filter out spurious random matches.
5. Anomaly Footprint:
   Projects matched inlier keypoints into a spatial anomaly density heatmap A_CM(x, y).
"""

import cv2
import numpy as np
from src.spatial.linear_filters import compute_local_variance
from src.spatial.transforms import to_grayscale


def extract_features_with_variance_gating(
    img: np.ndarray,
    n_features: int = 1500,
    var_thresh: float = 10.0,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """
    Extract SIFT keypoints and descriptors, filtering out low-variance flat regions.
    """
    gray = to_grayscale(img)
    var_map = compute_local_variance(gray, ksize=7)

    # Initialize SIFT or fallback to ORB
    try:
        detector = cv2.SIFT_create(nfeatures=n_features)
    except AttributeError:
        detector = cv2.ORB_create(nfeatures=n_features)

    kps, descs = detector.detectAndCompute(gray, None)
    if descs is None or len(kps) == 0:
        return [], np.empty((0, 128), dtype=np.float32)

    # Filter keypoints by local variance
    valid_kps = []
    valid_descs = []
    h, w = gray.shape

    for i, kp in enumerate(kps):
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)
        if var_map[y, x] >= var_thresh:
            valid_kps.append(kp)
            valid_descs.append(descs[i])

    if len(valid_kps) == 0:
        return [], np.empty((0, descs.shape[1]), dtype=descs.dtype)

    return valid_kps, np.array(valid_descs)


def match_copy_move_features(
    keypoints: list[cv2.KeyPoint],
    descriptors: np.ndarray,
    ratio_thresh: float = 0.65,
    min_spatial_dist: float = 20.0,
) -> list[tuple[cv2.KeyPoint, cv2.KeyPoint]]:
    """
    Match keypoints within the same image using generalized 2NN distance ratio
    and spatial displacement constraints.
    """
    if len(keypoints) < 4 or descriptors is None or len(descriptors) < 4:
        return []

    # Use Brute-Force Matcher with k=3 nearest neighbors (including self)
    norm_type = cv2.NORM_L2 if descriptors.dtype == np.float32 else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm_type)
    knn_matches = bf.knnMatch(descriptors, descriptors, k=min(4, len(descriptors)))

    matched_pairs = []
    seen_pairs = set()

    for matches in knn_matches:
        if len(matches) < 2:
            continue
        # matches[0] is typically the query descriptor itself (distance 0)
        p0 = matches[0]
        for m in matches[1:]:
            q_idx, t_idx = p0.queryIdx, m.trainIdx
            if q_idx == t_idx:
                continue

            pt1 = keypoints[q_idx].pt
            pt2 = keypoints[t_idx].pt
            spatial_dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

            if spatial_dist < min_spatial_dist:
                continue

            pair_id = tuple(sorted([q_idx, t_idx]))
            if pair_id in seen_pairs:
                continue

            # Standard distance ratio test compared to next distinct neighbor
            if len(matches) >= 3 and matches[2].trainIdx != t_idx:
                next_dist = matches[2].distance
            else:
                next_dist = m.distance * 1.5

            if m.distance < ratio_thresh * (next_dist + 1e-8):
                matched_pairs.append((keypoints[q_idx], keypoints[t_idx]))
                seen_pairs.add(pair_id)

    return matched_pairs


def detect_copy_move(
    img: np.ndarray,
    n_features: int = 1500,
    var_thresh: float = 10.0,
    ratio_thresh: float = 0.65,
    min_spatial_dist: float = 20.0,
    min_inliers: int = 4,
) -> tuple[np.ndarray, list[tuple[tuple[float, float], tuple[float, float]]], bool]:
    """
    Complete Copy-Move detection pipeline with RANSAC geometric verification.

    Returns:
        Tuple of:
          - anomaly_heatmap: (H, W), float64 in [0.0, 1.0]
          - inlier_point_pairs: list of ((x1, y1), (x2, y2))
          - is_detected: True if inlier count exceeds threshold.
    """
    h, w = img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float64)

    kps, descs = extract_features_with_variance_gating(img, n_features, var_thresh)
    matched_kps = match_copy_move_features(kps, descs, ratio_thresh, min_spatial_dist)

    if len(matched_kps) < min_inliers:
        return heatmap, [], False

    pts1 = np.float32([kp1.pt for kp1, kp2 in matched_kps])
    pts2 = np.float32([kp2.pt for kp1, kp2 in matched_kps])

    # RANSAC Affine Model Estimation
    model, inliers = cv2.estimateAffine2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0
    )

    inlier_pairs = []
    if inliers is not None and np.sum(inliers) >= min_inliers:
        inlier_mask = inliers.ravel().astype(bool)
        for i, is_inlier in enumerate(inlier_mask):
            if is_inlier:
                pt1 = (float(pts1[i][0]), float(pts1[i][1]))
                pt2 = (float(pts2[i][0]), float(pts2[i][1]))
                inlier_pairs.append((pt1, pt2))

                # Splat Gaussian density onto anomaly heatmap
                cv2.circle(heatmap, (int(pt1[0]), int(pt1[1])), 15, (0.8,), -1)
                cv2.circle(heatmap, (int(pt2[0]), int(pt2[1])), 15, (1.0,), -1)

        # Smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        max_v = np.max(heatmap)
        if max_v > 0:
            heatmap /= max_v
        return heatmap, inlier_pairs, True

    return heatmap, [], False
