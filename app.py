"""
S.P.E.C.T.R.A. — Streamlit Dashboard
Rekoras Crypto Analytics Dark Design System
Pure UI orchestrator (<= 200 lines). All DIP logic lives in src/.
"""
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from src.compression import compute_block_artifact_grid, compute_ela, compute_local_entropy
from src.detection import detect_copy_move, detect_splice_anomalies
from src.frequency import apply_homomorphic_filter, compute_2d_fft, get_magnitude_spectrum
from src.morphology import refine_tamper_mask
from src.spatial import apply_clahe, compute_sobel_gradients, to_grayscale
from tests.synthetic_forger import (
    create_synthetic_copy_move, create_synthetic_splice, generate_synthetic_base,
)

# ── Page Config & CSS ──────────────────────────────────────────
st.set_page_config(page_title="S.P.E.C.T.R.A.", page_icon="⚡", layout="wide")
css = Path("assets/style.css")
if css.exists():
    st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────
H = lambda s: st.markdown(s, unsafe_allow_html=True)
SL = lambda t: H(f'<div class="section-label"><div class="label-line"></div><span>{t}</span></div>')
SD = lambda: H('<div class="sidebar-divider"></div>')
SLL = lambda t: H(f'<p class="sidebar-section-label">{t}</p>')

# ── Top Nav ────────────────────────────────────────────────────
H("""<div class="spectra-nav">
  <div class="nav-brand"><div class="nav-logo">S</div>
    <span class="nav-title">S.P.E.C.T.R.A.</span></div>
  <span class="nav-pill">Forensic Analytics Engine</span>
  <div class="nav-status"><div class="status-dot"></div>
    <span>SYSTEM NOMINAL &nbsp;|&nbsp; 100% Classical DIP</span></div>
</div>""")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    H('<div class="sidebar-header"><span class="rekoras-badge">Control Terminal</span></div>')
    SLL("Input Feed")
    input_mode = st.selectbox("feed", ["Synthetic Benchmark","Upload Image"], label_visibility="collapsed")
    if input_mode == "Upload Image":
        up = st.file_uploader("Image", type=["png","jpg","jpeg","bmp"])
        img_bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR) if up else None
        if img_bgr is None:
            st.info("Upload an image to begin."); st.stop()
    else:
        SLL("Scenario")
        case = st.selectbox("case", ["Copy-Move","Splice","Authentic"], label_visibility="collapsed")
        base = generate_synthetic_base(256, 256)
        if case == "Copy-Move":
            img_bgr, _ = create_synthetic_copy_move(base, source_box=(30,30,80,80), dest_pos=(140,140))
        elif case == "Splice":
            img_bgr, _ = create_synthetic_splice(base, donor_quality=55, splice_box=(40,40,80,80))
        else:
            img_bgr = base
    SD()
    SLL("Hyperparameters")
    ela_q   = st.slider("ELA  Q_ref",          50, 98,   90)
    homo_d0 = st.slider("Homomorphic  D₀",     10.0, 80.0, 30.0)
    min_b   = st.slider("CCA Min Area (px)",   50, 500, 150)
    SD()
    st.button("⚡ EXECUTE ANALYSIS")

# ── Shared Computations ────────────────────────────────────────
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray    = to_grayscale(img_bgr)
h, w    = img_bgr.shape[:2]
mean_px = float(np.mean(gray)); std_px = float(np.std(gray))
_, ela_raw = compute_ela(img_bgr, quality=ela_q)
ela_score  = float(np.mean(ela_raw) * 100)

# ── Hero ───────────────────────────────────────────────────────
H("""<div class="spectra-hero">
  <div class="hero-label">Spatial Processing & Entropy Checks for Tamper Recognition Analysis</div>
  <h1>S.P.E.C.T.R.A.</h1>
  <p class="hero-sub">Classical DIP Forensics Engine &nbsp;·&nbsp; Gonzalez &amp; Woods</p>
</div>""")

# ── KPI Row ────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
kpis = [
    (k1,"Resolution",    f"{w}×{h}",         "cyan",  "px · sRGB uint8"),
    (k2,"Mean Luminance",f"{mean_px:.1f}",   "",      f"σ = {std_px:.1f}"),
    (k3,"ELA Score",     f"{ela_score:.3f}%","cyan" if ela_score<1 else "red",f"Q_ref = {ela_q}"),
    (k4,"DL Imports",    "0",                "green", "100% Classical DIP"),
]
for col, lbl, val, cls, sub in kpis:
    col.markdown(f'<div class="stat-card"><div class="stat-label">{lbl}</div>'
                 f'<div class="stat-value {cls}">{val}</div>'
                 f'<div class="stat-sub">{sub}</div></div>', unsafe_allow_html=True)
H("<br>")

# ── Tabs ───────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs(["🖼  Overview","🔬  Spatial","〰  Frequency","📦  Compression","🎯  Tamper"])

with t1:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        SL("Input Matrix")
        st.image(img_rgb, use_container_width=True, caption=f"f(x,y) — {w}×{h} px")
    with c2:
        SL("System Telemetry")
        clr = 'var(--crimson)' if ela_score > 1 else 'var(--emerald)'
        H(f"""<div class="rekoras-card">
          <p style="margin:0 0 .8rem;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text-2)">Pipeline Architecture</p>
          <p style="margin:.35rem 0">📐 <strong>Resolution:</strong> <span class="mono-tag">{w}×{h} px</span></p>
          <p style="margin:.35rem 0">📊 <strong>Signal Energy:</strong> <span class="mono-tag">μ={mean_px:.2f} σ={std_px:.2f}</span></p>
          <p style="margin:.35rem 0">🧠 <strong>Algorithm:</strong> <span class="mono-tag">100% Classical DIP</span></p>
          <p style="margin:.35rem 0">⚡ <strong>ELA Score:</strong> <span class="mono-tag" style="color:{clr}">{ela_score:.4f}%</span></p>
          <p style="margin:.35rem 0">🚫 <strong>DL Imports:</strong> <span class="mono-tag" style="color:var(--emerald)">0</span></p>
        </div>""")

with t2:
    SL("Spatial Transforms — Gonzalez & Woods Ch. 3")
    s1, s2, s3 = st.columns(3, gap="medium")
    s1.image(gray, caption="Luminance I(x,y)", use_container_width=True)
    s2.image(apply_clahe(img_bgr, clip_limit=2.5), caption="CLAHE Local Contrast", use_container_width=True)
    _, _, mag = compute_sobel_gradients(gray)
    s3.image(mag, caption="Sobel Gradient M(x,y)", use_container_width=True)

with t3:
    SL("Frequency Domain — Gonzalez & Woods Ch. 4")
    f_shift, _ = compute_2d_fft(gray)
    _, homo_res = apply_homomorphic_filter(gray, d0=homo_d0)
    f1, f2 = st.columns(2, gap="medium")
    f1.image(get_magnitude_spectrum(f_shift), caption="2D FFT Spectrum S(u,v)", use_container_width=True)
    f2.image(homo_res, caption="Homomorphic Residual Δ_illum(x,y)", use_container_width=True)

with t4:
    SL("JPEG Redundancy & Entropy — Ch. 8")
    ela_vis, _ = compute_ela(img_bgr, quality=ela_q)
    bag = compute_block_artifact_grid(img_bgr, block_size=8)
    entropy_map = compute_local_entropy(gray, radius=4)
    e1, e2, e3 = st.columns(3, gap="medium")
    e1.image(ela_vis, caption="Error Level Analysis E(x,y)", use_container_width=True)
    e2.image(bag, caption="Block Artifact Grid BAG(x,y)", use_container_width=True)
    e3.image(entropy_map, caption="Shannon Entropy H(x,y)", use_container_width=True)

with t5:
    SL("Forensic Detection — Ch. 9 & 10")
    cm_map, inliers, is_cm = detect_copy_move(img_bgr, n_features=1200)
    splice_map, _ = detect_splice_anomalies(img_bgr, ela_quality=ela_q, homomorphic_d0=homo_d0)
    active_map = cm_map if (is_cm and np.max(cm_map) > 0.3) else splice_map
    clean_mask, _ = refine_tamper_mask(active_map, min_area=min_b)
    overlay = img_rgb.copy(); overlay[clean_mask == 255] = [255, 51, 102]
    blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    r1, r2, r3 = st.columns(3, gap="medium")
    r1.image(active_map, caption="Anomaly Heatmap A(x,y)", use_container_width=True)
    r2.image(clean_mask, caption="Tamper Mask g(x,y)", use_container_width=True)
    r3.image(blended, caption="Forensic Overlay", use_container_width=True)
    H("<br>")
    pct = (np.count_nonzero(clean_mask) / clean_mask.size) * 100
    method = 'Copy-Move (Keypoint RANSAC)' if is_cm else 'Splice (ELA+Homo+BAG)'
    if pct > 0.5:
        H(f"""<div class="verdict-tampered"><div class="verdict-icon">⚠️</div><div>
          <p class="verdict-title">COMPLIANCE VIOLATION DETECTED</p>
          <p class="verdict-body">Flagged area: <strong style="color:var(--crimson)">{pct:.2f}%</strong>
          &nbsp;·&nbsp; Inliers: <strong style="color:var(--cyan-400)">{len(inliers)}</strong>
          &nbsp;·&nbsp; Method: <strong>{method}</strong></p></div></div>""")
    else:
        H("""<div class="verdict-authentic"><div class="verdict-icon">✅</div><div>
          <p class="verdict-title">ASSET INTEGRITY VERIFIED — AUTHENTIC</p>
          <p class="verdict-body">No statistically significant cloning or illumination splicing anomalies detected.</p>
          </div></div>""")
