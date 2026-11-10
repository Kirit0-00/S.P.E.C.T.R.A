"""
S.P.E.C.T.R.A. - Command Line Interface (CLI) Engine.
Headless forensic analysis runner for automated grading and batch execution (< 200 lines).
"""

import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np

from src.detection.copy_move import detect_copy_move
from src.detection.splice_detector import detect_splice_anomalies
from src.morphology.segmentation import refine_tamper_mask
from tests.synthetic_forger import (
    create_synthetic_copy_move,
    create_synthetic_splice,
    generate_synthetic_base,
)


def run_forensic_pipeline(
    img: np.ndarray,
    mode: str = "full",
    ela_q: int = 90,
    homo_d0: float = 30.0,
    min_area: int = 150,
) -> dict:
    """Execute complete classical DIP forensics pipeline."""
    t0 = time.perf_counter()
    h, w = img.shape[:2]

    cm_map, inliers, is_cm = np.zeros((h, w)), [], False
    splice_map, cues = np.zeros((h, w)), {}

    if mode in ("copy_move", "full"):
        cm_map, inliers, is_cm = detect_copy_move(img, n_features=1200)

    if mode in ("splice", "full"):
        splice_map, cues = detect_splice_anomalies(img, ela_quality=ela_q, homomorphic_d0=homo_d0)

    # Multi-modal fusion decision
    if mode == "copy_move":
        active_map = cm_map
    elif mode == "splice":
        active_map = splice_map
    else:
        active_map = cm_map if (is_cm and np.max(cm_map) > 0.3) else splice_map

    clean_mask, boundary = refine_tamper_mask(active_map, min_area=min_area)
    elapsed = time.perf_counter() - t0

    tamper_pixels = int(np.count_nonzero(clean_mask))
    tamper_ratio = float(tamper_pixels / clean_mask.size)

    return {
        "status": "TAMPERED" if tamper_ratio > 0.005 else "AUTHENTIC",
        "mode": mode,
        "resolution": [w, h],
        "tamper_pixels": tamper_pixels,
        "tamper_percentage": round(tamper_ratio * 100.0, 2),
        "inliers_detected": len(inliers),
        "runtime_seconds": round(elapsed, 4),
        "_clean_mask": clean_mask,
        "_active_map": active_map,
    }


def main():
    parser = argparse.ArgumentParser(description="S.P.E.C.T.R.A. Classical DIP Image Forensics CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--mode", choices=["copy_move", "splice", "full"], default="full")
    parser.add_argument("--synthetic", choices=["copy_move", "splice", "authentic"], default=None)
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save output masks")
    parser.add_argument("--json", action="store_true", help="Print output formatted as JSON")
    args = parser.parse_args()

    # Load input or synthesize
    if args.image:
        img_bgr = cv2.imread(args.image)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image at {args.image}")
    else:
        base = generate_synthetic_base(256, 256)
        if args.synthetic == "copy_move":
            img_bgr, _ = create_synthetic_copy_move(base, source_box=(30, 30, 80, 80), dest_pos=(140, 140))
        elif args.synthetic == "splice":
            img_bgr, _ = create_synthetic_splice(base, donor_quality=55, splice_box=(40, 40, 80, 80))
        else:
            img_bgr = base

    results = run_forensic_pipeline(img_bgr, mode=args.mode)

    # Save artifacts
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = results.pop("_clean_mask")
    heatmap = results.pop("_active_map")

    cv2.imwrite(str(out_dir / "binary_tamper_mask.png"), mask)
    cv2.imwrite(str(out_dir / "anomaly_heatmap.png"), (heatmap * 255).astype(np.uint8))

    # Colored overlay
    overlay = img_bgr.copy()
    overlay[mask == 255] = [45, 45, 255]
    blended = cv2.addWeighted(img_bgr, 0.65, overlay, 0.35, 0)
    cv2.imwrite(str(out_dir / "tamper_overlay.png"), blended)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n=================== S.P.E.C.T.R.A. FORENSIC REPORT ===================")
        print(f" Status:             {results['status']}")
        print(f" Analyzed Dimension: {results['resolution'][0]} x {results['resolution'][1]} px")
        print(f" Tamper Percentage:  {results['tamper_percentage']}%")
        print(f" Keypoint Inliers:   {results['inliers_detected']}")
        print(f" Pipeline Runtime:   {results['runtime_seconds']}s")
        print(f" Artifacts Saved To: {out_dir.resolve()}/")
        print(f"======================================================================\n")


if __name__ == "__main__":
    main()
