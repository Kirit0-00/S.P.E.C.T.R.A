"""
S.P.E.C.T.R.A. - Streamlit Interactive Dashboard UI.
Frontend: Inspired by Rekoras (Crypto Compliance Analytics) & Uiverse.io 3D Tactile Buttons.
Architecture: Pure UI orchestrator (< 200 lines). Core DIP algorithms from src/ modules.
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
    create_synthetic_copy_move,
    create_synthetic_splice,
    generate_synthetic_base,
)

st.set_page_config(page_title="S.P.E.C.T.R.A. // Forensic Analytics", page_icon="⚡", layout="wide")

# Inject Rekoras Design System & 3D Tactile Buttons
css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Rekoras Top Header
st.markdown(
    """<div style='text-align: center; padding: 1.2rem 0 1.8rem;'>
      <span class='rekoras-badge'>FORENSIC COMPLIANCE & TAMPER VERIFICATION</span>
      <h1 style='font-size: 2.8rem; margin: 0.5rem 0 0.2rem; background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 60%, #10B981 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>S.P.E.C.T.R.A.</h1>
      <p style='color: #94A3B8; font-size: 1.05rem; margin-top: 0;'>Spatial Processing and Entropy Checks for Tamper Recognition Analysis</p>
    </div>""",
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.markdown("<h2 style='color: #00F2FE;'>⚙️ Control Terminal</h2>", unsafe_allow_html=True)
input_mode = st.sidebar.selectbox("Input Feed", ["Synthetic Benchmark Scenarios", "Upload Live Image"])

if input_mode == "Upload Live Image":
    uploaded = st.sidebar.file_uploader("Upload Image (PNG, JPG, BMP)", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        st.info("Awaiting input stream. Upload an image or select a benchmark scenario.")
        st.stop()
else:
    test_case = st.sidebar.selectbox("Select Benchmark Case", ["Copy-Move (Cloning)", "Splice (Dual Compression)", "Authentic Baseline"])
    base = generate_synthetic_base(256, 256)
    if test_case == "Copy-Move (Cloning)":
        img_bgr, _ = create_synthetic_copy_move(base, source_box=(30, 30, 80, 80), dest_pos=(140, 140))
    elif test_case == "Splice (Dual Compression)":
        img_bgr, _ = create_synthetic_splice(base, donor_quality=55, splice_box=(40, 40, 80, 80))
    else:
        img_bgr = base

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = to_grayscale(img_bgr)

# Interactive Forensic Parameter Sliders
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='color: #4FACFE; font-weight: 700;'>Hyperparameter Calibration</p>", unsafe_allow_html=True)
ela_q = st.sidebar.slider("ELA Reference Quality Q_ref", 50, 98, 90)
homo_d0 = st.sidebar.slider("Homomorphic Cutoff Frequency D0", 10.0, 80.0, 30.0)
min_blob = st.sidebar.slider("Morphological Cluster Gating (CCA)", 50, 500, 150)

# Tactical Action Trigger
st.sidebar.markdown("---")
st.sidebar.button("⚡ EXECUTE DEEP ANALYSIS")

# Analytical Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Telemetry", "🌐 Spatial Domain", "⚡ Frequency Domain", "📦 Entropy & ELA", "🎯 Tamper Localization"
])

with tab1:
    c1, c2 = st.columns([1, 1])
    c1.image(img_rgb, caption=f"Analyzed Matrix [{img_bgr.shape[1]}x{img_bgr.shape[0]}]", use_container_width=True)
    c2.markdown(
        f"""<div class='rekoras-card'>
          <h3 style='color: #00F2FE; margin-top: 0;'>Input Telemetry</h3>
          <p><strong>Resolution:</strong> <code>{img_bgr.shape[1]} × {img_bgr.shape[0]} px</code></p>
          <p><strong>Color Space:</strong> <code>sRGB 3-Channel (uint8)</code></p>
          <p><strong>Mean Signal Energy:</strong> <code>{np.mean(gray):.2f}</code></p>
          <p><strong>Theoretical Architecture:</strong> <code>100% Classical DIP (Gonzalez & Woods)</code></p>
          <p><strong>Deep Learning Imports:</strong> <code style='color: #10B981;'>0% (Fully Interpretable)</code></p>
        </div>""",
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown("<p style='color: #00F2FE; font-weight: 700;'>Spatial Transforms & Boundary Discontinuities (Ch. 3)</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.image(gray, caption="Grayscale Luminance I(x, y)", use_container_width=True)
    c2.image(apply_clahe(img_bgr, clip_limit=2.5), caption="CLAHE Local Contrast Enhancement", use_container_width=True)
    _, _, mag = compute_sobel_gradients(gray)
    c3.image(mag, caption="Sobel Spatial Gradient Magnitude M(x, y)", use_container_width=True)

with tab3:
    st.markdown("<p style='color: #00F2FE; font-weight: 700;'>2D Fourier Spectrum & Homomorphic Illumination Invariance (Ch. 4)</p>", unsafe_allow_html=True)
    f_shift, _ = compute_2d_fft(gray)
    mag_spectrum = get_magnitude_spectrum(f_shift)
    _, homo_res = apply_homomorphic_filter(gray, d0=homo_d0)
    c1, c2 = st.columns(2)
    c1.image(mag_spectrum, caption="Centered 2D FFT Log Magnitude S(u, v)", use_container_width=True)
    c2.image(homo_res, caption="Homomorphic Illumination Residual Δ_illum(x, y)", use_container_width=True)

with tab4:
    st.markdown("<p style='color: #00F2FE; font-weight: 700;'>Quantization Redundancy & Information Entropy (Ch. 8)</p>", unsafe_allow_html=True)
    ela_vis, _ = compute_ela(img_bgr, quality=ela_q)
    bag = compute_block_artifact_grid(img_bgr, block_size=8)
    entropy_map = compute_local_entropy(gray, radius=4)
    c1, c2, c3 = st.columns(3)
    c1.image(ela_vis, caption="Error Level Analysis (ELA Matrix)", use_container_width=True)
    c2.image(bag, caption="8x8 Block Artifact Grid (BAG Discontinuity)", use_container_width=True)
    c3.image(entropy_map, caption="Local Shannon Rank Entropy H(x, y)", use_container_width=True)

with tab5:
    st.markdown("<p style='color: #00F2FE; font-weight: 700;'>Tamper Localization & Morphological Refinement (Ch. 9 & 10)</p>", unsafe_allow_html=True)
    cm_map, inliers, is_cm = detect_copy_move(img_bgr, n_features=1200)
    splice_map, _ = detect_splice_anomalies(img_bgr, ela_quality=ela_q, homomorphic_d0=homo_d0)

    active_map = cm_map if (is_cm and np.max(cm_map) > 0.3) else splice_map
    clean_mask, boundary = refine_tamper_mask(active_map, min_area=min_blob)

    overlay = img_rgb.copy()
    overlay[clean_mask == 255] = [255, 45, 85]  # Neon crimson highlight
    blended_overlay = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)

    c1, c2, c3 = st.columns(3)
    c1.image(active_map, caption="Raw Multi-Modal Anomaly Response A(x, y)", use_container_width=True)
    c2.image(clean_mask, caption="Refined Binary Tamper Mask g(x, y)", use_container_width=True)
    c3.image(blended_overlay, caption="Forensic Verification Overlay", use_container_width=True)

    tamper_pct = (np.count_nonzero(clean_mask) / clean_mask.size) * 100
    if tamper_pct > 0.5:
        st.markdown(f"""<div class='rekoras-card' style='border-color: rgba(239, 68, 68, 0.5);'>
          <h4 style='color: #EF4444; margin: 0;'>⚠️ COMPLIANCE VIOLATION DETECTED</h4>
          <p style='margin: 0.3rem 0 0;'>Flagged Tamper Area: <strong>{tamper_pct:.2f}%</strong> | RANSAC Verified Inliers: <strong>{len(inliers)}</strong></p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='rekoras-card' style='border-color: rgba(16, 185, 129, 0.5);'>
          <h4 style='color: #10B981; margin: 0;'>✅ ASSET INTEGRITY VERIFIED (AUTHENTIC)</h4>
          <p style='margin: 0.3rem 0 0;'>No statistically significant geometric cloning or illumination splicing anomalies detected.</p>
        </div>""", unsafe_allow_html=True)
