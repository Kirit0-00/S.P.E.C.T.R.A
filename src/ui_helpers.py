"""
S.P.E.C.T.R.A. — UI Helper Module
Houses experiment image registry, annotation graphics, and DIP educational cards.
Pure utility functions (<= 200 lines). Zero DL dependencies.
"""
from pathlib import Path
import cv2
import numpy as np

EXPERIMENT_SAMPLES = {
    "01_ant_macro": {
        "title": "🐜 Macro Ant Photo (Canon DSLR)",
        "desc": "High-texture biological macro photograph with authentic camera sensor noise.",
        "real": "data/experiments/real/01_ant_macro.jpg",
        "copymove": "data/experiments/forged_copymove/01_ant_macro_copymove.jpg",
        "splice": "data/experiments/forged_splice/01_ant_macro_splice.jpg",
        "cm_detail": "120×150 px ant body patch cloned into background region (~268 px displacement).",
        "spl_detail": "27,000 px patch re-compressed at Q=35 and brightened 1.5×."
    },
    "02_colosseum": {
        "title": "🏛️ Colosseum (Architectural Scene)",
        "desc": "Complex repetitive stone arches, perspective geometry, and open sky.",
        "real": "data/experiments/real/02_colosseum.jpg",
        "copymove": "data/experiments/forged_copymove/02_colosseum_copymove.jpg",
        "splice": "data/experiments/forged_splice/02_colosseum_splice.jpg",
        "cm_detail": "Archway structure cloned across the facade to simulate architectural duplication.",
        "spl_detail": "Upper structural region replaced with low-quality Q=30 patch with +15 additive bias."
    },
    "03_sand_dunes": {
        "title": "🏜️ Sand Dunes (Landscape Gradient)",
        "desc": "Natural lighting illumination gradients and undulating sand textures.",
        "real": "data/experiments/real/03_sand_dunes.jpg",
        "copymove": "data/experiments/forged_copymove/03_sand_dunes_copymove.jpg",
        "splice": "data/experiments/forged_splice/03_sand_dunes_splice.jpg",
        "cm_detail": "Dune ridge line cloned across the valley to alter desert topography.",
        "spl_detail": "Sky-ridge boundary spliced with Q=35 donor under 1.5× illuminance boost."
    },
    "04_sparrow": {
        "title": "🐦 Sparrow (Wildlife & Plumage)",
        "desc": "High-frequency feather textures and soft natural bokeh background.",
        "real": "data/experiments/real/04_sparrow.jpg",
        "copymove": "data/experiments/forged_copymove/04_sparrow_copymove.jpg",
        "splice": "data/experiments/forged_splice/04_sparrow_splice.jpg",
        "cm_detail": "Wing/plumage patch cloned to the right quadrant (~200 px offset).",
        "spl_detail": "Central subject background replaced with foreign patch compressed at Q=35."
    }
}

def load_experiment_sample(sample_id: str, variant: str) -> tuple[np.ndarray | None, str]:
    """Loads specified experiment image and returns (bgr_image, detail_text)."""
    meta = EXPERIMENT_SAMPLES.get(sample_id, EXPERIMENT_SAMPLES["01_ant_macro"])
    path_map = {
        "Authentic": (meta["real"], f"Authentic Asset: {meta['desc']}"),
        "Copy-Move": (meta["copymove"], f"Copy-Move Tampering: {meta['cm_detail']}"),
        "Splice": (meta["splice"], f"Splice Tampering: {meta['spl_detail']}")
    }
    fpath, detail = path_map.get(variant, path_map["Authentic"])
    img = cv2.imread(fpath)
    return img, detail

def draw_copy_move_vectors(img_rgb: np.ndarray, inlier_pairs: list) -> np.ndarray:
    """Draws cyan and emerald connected vectors between matched cloned keypoint pairs."""
    canvas = img_rgb.copy()
    if not inlier_pairs:
        return canvas
    for (x1, y1), (x2, y2) in inlier_pairs:
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.line(canvas, p1, p2, (0, 242, 254), 2, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, (0, 255, 157), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 4, (255, 51, 102), -1, cv2.LINE_AA)
    return canvas

def make_dip_card(title: str, formula: str, principle: str, indicator: str) -> str:
    """Returns styled HTML card explaining DIP math, concept, and visual cues."""
    return f"""<div class="rekoras-card" style="padding:1rem;margin-bottom:0.8rem">
      <div style="font-weight:700;color:var(--cyan-400);font-size:0.82rem;margin-bottom:0.3rem">{title}</div>
      <div style="font-family:var(--mono);color:var(--text-1);font-size:0.75rem;background:rgba(0,0,0,0.3);padding:0.25rem 0.5rem;border-radius:4px;margin-bottom:0.4rem">{formula}</div>
      <div style="font-size:0.75rem;color:var(--text-2);margin-bottom:0.3rem"><strong>DIP Principle:</strong> {principle}</div>
      <div style="font-size:0.72rem;color:var(--azure)"><strong>Look for:</strong> {indicator}</div>
    </div>"""
