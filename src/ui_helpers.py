"""
S.P.E.C.T.R.A. — UI Helper Module
Houses experiment image registry, DIP syllabus matrix, and educational cards.
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

SYLLABUS_MODULES = [
    ("Unit 1", "Spatial Domain Processing", "Ch. 3", "I = 0.299R+0.587G+0.114B; M = √(Gx²+Gy²); CLAHE", "Extracts luminance, exposes subtle contrast seams, detects edge step jumps.", "src/spatial/"),
    ("Unit 2", "Frequency Domain Analysis", "Ch. 4", "S = ln(1+|F|); ln(f) = ln(i)+ln(r) -> HPF -> exp", "2D FFT centered spectrum shows periodic repetition; Homomorphic isolates illumination mismatch.", "src/frequency/"),
    ("Unit 3", "Compression & Entropy", "Ch. 8", "E = |f - f_Q90|·α; BAG = [1,-2,1]; H = -Σ p log2 p", "ELA reveals quantization equilibrium; BAG catches 8×8 grid offset; Entropy maps randomness.", "src/compression/"),
    ("Unit 4", "Feature Matching & Fusion", "Ch. 11", "d(k1,k2)/d(k1,k3) < 0.65; p' = H·p; A = 0.5E+0.3H+0.2B", "SIFT+g2NN finds cloned descriptors; RANSAC verifies geometry; Anomaly fusion combines cues.", "src/detection/"),
    ("Unit 5", "Morphology & Segmentation", "Ch. 9-10", "A∘B = (A⊖B)⊕B; Otsu σB²(t); CCA Area ≥ τ_min", "Otsu binarization, geodesic hole-filling, opening/closing, and connected component gating.", "src/morphology/"),
]

def load_experiment_sample(sample_id: str, variant: str) -> tuple[np.ndarray | None, str]:
    meta = EXPERIMENT_SAMPLES.get(sample_id, EXPERIMENT_SAMPLES["01_ant_macro"])
    path_map = {
        "Authentic": (meta["real"], f"Authentic Asset: {meta['desc']}"),
        "Copy-Move": (meta["copymove"], f"Copy-Move Tampering: {meta['cm_detail']}"),
        "Splice": (meta["splice"], f"Splice Tampering: {meta['spl_detail']}")
    }
    fpath, detail = path_map.get(variant, path_map["Authentic"])
    return cv2.imread(fpath), detail

def draw_copy_move_vectors(img_rgb: np.ndarray, inlier_pairs: list) -> np.ndarray:
    canvas = img_rgb.copy()
    if not inlier_pairs:
        return canvas
    for (x1, y1), (x2, y2) in inlier_pairs:
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.line(canvas, p1, p2, (0, 242, 254), 2, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, (0, 255, 157), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 4, (255, 51, 102), -1, cv2.LINE_AA)
    return canvas

def make_dip_card(unit: str, title: str, formula: str, principle: str, indicator: str) -> str:
    return f"""<div class="rekoras-card" style="padding:0.9rem;margin-bottom:0.75rem">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem">
        <span style="font-weight:700;color:var(--cyan-400);font-size:0.82rem">{title}</span>
        <span class="mono-tag" style="font-size:0.68rem;padding:0.1rem 0.4rem">{unit}</span>
      </div>
      <div style="font-family:var(--mono);color:var(--text-1);font-size:0.73rem;background:rgba(0,0,0,0.35);padding:0.25rem 0.5rem;border-radius:4px;margin-bottom:0.35rem">{formula}</div>
      <div style="font-size:0.74rem;color:var(--text-2);margin-bottom:0.25rem"><strong>DIP Principle:</strong> {principle}</div>
      <div style="font-size:0.72rem;color:var(--azure)"><strong>Forensic Clue:</strong> {indicator}</div>
    </div>"""

def render_syllabus_matrix() -> str:
    rows = "".join([
        f"""<tr style="border-bottom:1px solid rgba(0,242,254,0.1)">
          <td style="padding:0.6rem 0.8rem;font-weight:700;color:var(--cyan-400)">{u}</td>
          <td style="padding:0.6rem 0.8rem;color:var(--text-1)"><strong>{top}</strong><br><span style="font-size:0.72rem;color:var(--text-3)">Gonzalez & Woods {ch}</span></td>
          <td style="padding:0.6rem 0.8rem;font-family:var(--mono);font-size:0.72rem;color:var(--text-2)">{form}</td>
          <td style="padding:0.6rem 0.8rem;font-size:0.75rem;color:var(--text-2)">{app}</td>
          <td style="padding:0.6rem 0.8rem;font-family:var(--mono);font-size:0.72rem;color:var(--emerald)">{mod}</td>
        </tr>""" for u, top, ch, form, app, mod in SYLLABUS_MODULES
    ])
    return f"""<div class="rekoras-card" style="padding:1.2rem;overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:0.78rem">
        <thead>
          <tr style="border-bottom:2px solid rgba(0,242,254,0.3);text-align:left;color:var(--text-1)">
            <th style="padding:0.5rem 0.8rem">Course Unit</th>
            <th style="padding:0.5rem 0.8rem">DIP Syllabus Topic</th>
            <th style="padding:0.5rem 0.8rem">Core Formula / Math</th>
            <th style="padding:0.5rem 0.8rem">How SPECTRA Uses It</th>
            <th style="padding:0.5rem 0.8rem">Source Module</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""
