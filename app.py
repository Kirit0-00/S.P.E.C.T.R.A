"""
S.P.E.C.T.R.A. - Streamlit Interactive Dashboard UI.
Spatial Processing and Entropy Checks for Tamper Recognition Analysis.
Pure UI orchestrator (< 200 lines). Core science imported from src/ modules.
"""

import cv2
import numpy as np
import streamlit as st

from src.compression import compute_block_artifact_grid, compute_ela, compute_local_entropy
from src.detection import detect_copy_move, detect_splice_anomalies
from src.frequency import apply_homomorphic_filter, compute_2d_fft, get_magnitude_spectrum
from src.morphology import refine_tamper_mask
from src.spatial import apply_clahe, apply_laplacian, compute_sobel_gradients, to_grayscale
from tests.synthetic_forger import (
    create_synthetic_copy_move,
    create_synthetic_splice,
    generate_synthetic_base,
)

st.set_page_config(page_title="S.P.E.C.T.R.A. Forensics", page_icon="🔍", layout="wide")

st.markdown(
    """<h1 style='text-align: center; color: #58a6ff;'>🔍 S.P.E.C.T.R.A.</h1>
    <p style='text-align: center; font-size: 1.15rem; color: #8b949e;'>
    <strong>Spatial Processing and Entropy Checks for Tamper Recognition Analysis</strong><br>
    <em>Classical Digital Image Processing (Gonzalez & Woods) Forensics Suite</em></p>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.header("⚙️ Pipeline Configuration")
input_mode = st.sidebar.selectbox("Input Source", ["Synthetic Forgery Testbeds", "Upload Image"])

if input_mode == "Upload Image":
    uploaded = st.sidebar.file_uploader("Upload Image (PNG, JPG, BMP)", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        st.info("Please upload an image or switch to Synthetic Forgery Testbeds in the sidebar.")
        st.stop()
else:
    test_case = st.sidebar.selectbox("Select Testbed Scenario", ["Copy-Move (Cloning)", "Splice (Dual Compression)", "Authentic Scene"])
    base = generate_synthetic_base(256, 256)
    if test_case == "Copy-Move (Cloning)":
        img_bgr, _ = create_synthetic_copy_move(base, source_box=(30, 30, 80, 80), dest_pos=(140, 140))
    elif test_case == "Splice (Dual Compression)":
        img_bgr, _ = create_synthetic_splice(base, donor_quality=55, splice_box=(40, 40, 80, 80))
    else:
        img_bgr = base

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = to_grayscale(img_bgr)

# Forensic Parameter Sliders
ela_q = st.sidebar.slider("ELA Quality Q_ref", 50, 98, 90)
homo_d0 = st.sidebar.slider("Homomorphic Cutoff D0", 10.0, 80.0, 30.0)
min_blob = st.sidebar.slider("Min Tamper Blob Area (CCA)", 50, 500, 150)

# Layout Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖼️ Overview", "🌐 Spatial Domain", "⚡ Frequency Domain", "📦 Compression & Entropy", "🎯 Tamper Localization"
])

with tab1:
    col1, col2 = st.columns([1, 1])
    col1.image(img_rgb, caption=f"Analyzed Image ({img_bgr.shape[1]}x{img_bgr.shape[0]})", use_container_width=True)
    col2.markdown("### Image Attributes\n"
                  f"- **Dimensions:** {img_bgr.shape[0]} × {img_bgr.shape[1]} px\n"
                  f"- **Channels:** {img_bgr.shape[2]} (RGB)\n"
                  f"- **Mean Luminance:** {np.mean(gray):.2f}\n"
                  "- **Zero Deep Learning:** 100% Classical DIP Pipeline")

with tab2:
    st.subheader("Spatial Domain Transforms & Filtering (Gonzalez & Woods Ch. 3)")
    c1, c2, c3 = st.columns(3)
    c1.image(gray, caption="Grayscale Luminance I(x, y)", use_container_width=True)
    c2.image(apply_clahe(img_bgr, clip_limit=2.5), caption="CLAHE Contrast Enhanced", use_container_width=True)
    _, _, mag = compute_sobel_gradients(gray)
    c3.image(mag, caption="Sobel Gradient Magnitude M(x, y)", use_container_width=True)

with tab3:
    st.subheader("2D Fast Fourier Transform & Homomorphic Filtering (Ch. 4)")
    f_shift, _ = compute_2d_fft(gray)
    mag_spectrum = get_magnitude_spectrum(f_shift)
    _, homo_res = apply_homomorphic_filter(gray, d0=homo_d0)
    c1, c2 = st.columns(2)
    c1.image(mag_spectrum, caption="Centered 2D FFT Magnitude Spectrum S(u, v)", use_container_width=True)
    c2.image(homo_res, caption="Homomorphic Illumination Residual Δ_illum(x, y)", use_container_width=True)

with tab4:
    st.subheader("Compression & Information Entropy Checks (Ch. 8)")
    ela_vis, ela_raw = compute_ela(img_bgr, quality=ela_q)
    bag = compute_block_artifact_grid(img_bgr, block_size=8)
    entropy_map = compute_local_entropy(gray, radius=4)
    c1, c2, c3 = st.columns(3)
    c1.image(ela_vis, caption="Error Level Analysis (ELA Visual)", use_container_width=True)
    c2.image(bag, caption="8x8 Block Artifact Grid (BAG)", use_container_width=True)
    c3.image(entropy_map, caption="Local Shannon Entropy Map", use_container_width=True)

with tab5:
    st.subheader("Forensic Detection & Morphological Mask Refinement (Ch. 9 & 10)")
    cm_map, inliers, is_cm = detect_copy_move(img_bgr, n_features=1200)
    splice_map, _ = detect_splice_anomalies(img_bgr, ela_quality=ela_q, homomorphic_d0=homo_d0)

    active_map = cm_map if (is_cm and np.max(cm_map) > 0.3) else splice_map
    clean_mask, boundary = refine_tamper_mask(active_map, min_area=min_blob)

    # Composite Overlay
    overlay = img_rgb.copy()
    overlay[clean_mask == 255] = [255, 60, 60]  # Red tamper highlight
    blended_overlay = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)

    c1, c2, c3 = st.columns(3)
    c1.image(active_map, caption="Raw Forensic Anomaly Heatmap A(x, y)", use_container_width=True)
    c2.image(clean_mask, caption="Refined Binary Tamper Mask g(x, y)", use_container_width=True)
    c3.image(blended_overlay, caption="Forensic Overlay (Tamper Highlight)", use_container_width=True)

    tamper_percent = (np.count_nonzero(clean_mask) / clean_mask.size) * 100
    if tamper_percent > 0.5:
        st.error(f"⚠️ TAMPERING DETECTED: {tamper_percent:.2f}% of pixels flagged in forged region. (Inlier Pairs: {len(inliers)})")
    else:
        st.success("✅ AUTHENTIC: No significant tampering clusters detected.")
