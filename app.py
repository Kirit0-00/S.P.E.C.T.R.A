"""
S.P.E.C.T.R.A. — Rekoras Crypto Analytics Dark Forensic Dashboard
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
from src.ui_helpers import (
    EXPERIMENT_SAMPLES, draw_copy_move_vectors, load_experiment_sample, make_dip_card,
)
from tests.synthetic_forger import (
    create_synthetic_copy_move, create_synthetic_splice, generate_synthetic_base,
)

# ── Page Config & CSS ──────────────────────────────────────────
st.set_page_config(page_title="S.P.E.C.T.R.A.", page_icon="⚡", layout="wide")
css = Path("assets/style.css")
if css.exists():
    st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)

H = lambda s: st.markdown(s, unsafe_allow_html=True)
SL = lambda t: H(f'<div class="section-label"><div class="label-line"></div><span>{t}</span></div>')
SD = lambda: H('<div class="sidebar-divider"></div>')
SLL = lambda t: H(f'<p class="sidebar-section-label">{t}</p>')

# ── Top Nav ────────────────────────────────────────────────────
H("""<div class="spectra-nav"><div class="nav-brand"><div class="nav-logo">S</div>
  <span class="nav-title">S.P.E.C.T.R.A.</span></div>
  <span class="nav-pill">Forensic Analytics Engine</span>
  <div class="nav-status"><div class="status-dot"></div>
  <span>SYSTEM NOMINAL &nbsp;|&nbsp; 100% Classical DIP</span></div></div>""")

# ── Sidebar Feed Control ───────────────────────────────────────
with st.sidebar:
    H('<div class="sidebar-header"><span class="rekoras-badge">Control Terminal</span></div>')
    SLL("Input Feed Mode")
    feed = st.selectbox("Feed", ["🔬 Live Experiment Gallery", "Synthetic Benchmark", "Upload Image"], label_visibility="collapsed")
    context_desc = ""
    if feed == "🔬 Live Experiment Gallery":
        SLL("Select Real Scene")
        s_opts = {k: v["title"] for k, v in EXPERIMENT_SAMPLES.items()}
        chosen_key = st.selectbox("Scene", list(s_opts.keys()), format_func=lambda k: s_opts[k], label_visibility="collapsed")
        SLL("Asset Variant")
        variant = st.radio("Variant", ["Authentic", "Copy-Move", "Splice"], horizontal=True, label_visibility="collapsed")
        img_bgr, context_desc = load_experiment_sample(chosen_key, variant)
    elif feed == "Synthetic Benchmark":
        SLL("Scenario")
        scen = st.selectbox("Case", ["Copy-Move", "Splice", "Authentic"], label_visibility="collapsed")
        base = generate_synthetic_base(256, 256)
        if scen == "Copy-Move":
            img_bgr, _ = create_synthetic_copy_move(base, (30, 30, 80, 80), (140, 140))
        elif scen == "Splice":
            img_bgr, _ = create_synthetic_splice(base, donor_quality=55, splice_box=(40, 40, 80, 80))
        else:
            img_bgr = base
        context_desc = f"Procedural Synthetic Benchmark: {scen} Scenario"
    else:
        up = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"])
        img_bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR) if up else None
        if img_bgr is None:
            st.info("Upload an image to execute forensics."); st.stop()
        context_desc = "User Uploaded Image"
    SD(); SLL("Forensic Hyperparameters")
    ela_q = st.slider("ELA Q_ref", 50, 98, 90)
    homo_d0 = st.slider("Homomorphic D₀", 10.0, 80.0, 30.0)
    min_b = st.slider("CCA Min Area (px)", 50, 500, 150)
    cm_ratio = st.slider("g2NN Match Ratio", 0.50, 0.85, 0.65, 0.05)
    SD(); st.button("⚡ EXECUTE COMPLIANCE CHECK")

# ── Precomputations & Telemetry ────────────────────────────────
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = to_grayscale(img_bgr)
h, w = img_bgr.shape[:2]
mean_px, std_px = float(np.mean(gray)), float(np.std(gray))
_, ela_raw = compute_ela(img_bgr, quality=ela_q)
ela_score = float(np.mean(ela_raw) * 100)

# ── Hero & KPI Header ──────────────────────────────────────────
H("""<div class="spectra-hero">
  <div class="hero-label">Spatial Processing & Entropy Checks for Tamper Recognition Analysis</div>
  <h1>S.P.E.C.T.R.A.</h1>
  <p class="hero-sub">100% Classical DIP Engine &nbsp;·&nbsp; Gonzalez &amp; Woods Architecture</p></div>""")

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="stat-card"><div class="stat-label">Resolution</div><div class="stat-value cyan">{w}×{h}</div><div class="stat-sub">px · sRGB Matrix</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="stat-card"><div class="stat-label">Signal Energy</div><div class="stat-value">{mean_px:.1f}</div><div class="stat-sub">σ = {std_px:.1f} Luminance</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="stat-card"><div class="stat-label">ELA Anomaly</div><div class="stat-value {"cyan" if ela_score<1.0 else "red"}">{ela_score:.3f}%</div><div class="stat-sub">Q_ref = {ela_q}</div></div>', unsafe_allow_html=True)
k4.markdown('<div class="stat-card"><div class="stat-label">DL Dependencies</div><div class="stat-value green">0</div><div class="stat-sub">100% Classical DIP</div></div>', unsafe_allow_html=True)
H("<br>")

# ── Inspection Tabs ────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs(["🖼  Overview", "🔬  Spatial", "〰  Frequency", "📦  Compression", "🎯  Tamper Localization"])

with t1:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        SL("Asset Input Matrix f(x,y)")
        st.image(img_rgb, use_container_width=True, caption=f"Active Feed Matrix — {w}×{h} px")
    with c2:
        SL("Operational Context & Telemetry")
        clr = 'var(--crimson)' if ela_score > 1.0 else 'var(--emerald)'
        H(f"""<div class="rekoras-card">
          <p style="margin:0 0 .6rem;font-size:.72rem;font-weight:700;text-transform:uppercase;color:var(--text-2)">Asset Profile</p>
          <p style="margin:.3rem 0"><strong>Target Description:</strong> <span style="color:var(--cyan-400)">{context_desc}</span></p>
          <p style="margin:.3rem 0">📐 <strong>Spatial Resolution:</strong> <span class="mono-tag">{w}×{h} px</span></p>
          <p style="margin:.3rem 0">📊 <strong>Luminance Distribution:</strong> <span class="mono-tag">μ={mean_px:.2f}, σ={std_px:.2f}</span></p>
          <p style="margin:.3rem 0">⚡ <strong>Quantization Residual:</strong> <span class="mono-tag" style="color:{clr}">{ela_score:.4f}%</span></p>
          <p style="margin:.3rem 0">🛡️ <strong>Compliance Standard:</strong> <span class="mono-tag">Gonzalez & Woods Classical DIP</span></p>
        </div>""")

with t2:
    SL("Spatial Domain Filtering — Gonzalez & Woods Chapter 3")
    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        st.image(gray, caption="Luminance I(x,y)", use_container_width=True)
        H(make_dip_card("Luminance Decoupling", "I = 0.299R + 0.587G + 0.114B", "Removes chrominance bias; isolates perceived brightness.", "Uniform tonal representation across channels."))
    with s2:
        clahe_img = apply_clahe(img_bgr, clip_limit=2.5)
        st.image(clahe_img, caption="CLAHE Enhanced Image", use_container_width=True)
        H(make_dip_card("Adaptive Equalization (CLAHE)", "Tile-based Local Hist CDF Redistribution", "Amplifies contrast in localized 8×8 tiles while clipping noise peaks.", "Reveals subtle feathering and paste boundaries."))
    with s3:
        _, _, mag = compute_sobel_gradients(gray)
        st.image(mag, caption="Sobel Magnitude M(x,y)", use_container_width=True)
        H(make_dip_card("Sobel Directional Derivatives", "M(x,y) = sqrt(Gx² + Gy²)", "First-order spatial difference operators detect abrupt gradients.", "Sharp edge lines outlining pasted donor regions."))

with t3:
    SL("Frequency Domain & Illumination Analysis — Chapter 4")
    f_shift, _ = compute_2d_fft(gray)
    _, homo_res = apply_homomorphic_filter(gray, d0=homo_d0)
    f1, f2 = st.columns(2, gap="medium")
    with f1:
        st.image(get_magnitude_spectrum(f_shift), caption="2D Centered FFT Spectrum S(u,v)", use_container_width=True)
        H(make_dip_card("2D Centered Fourier Spectrum", "S(u,v) = ln(1 + |F(u,v)|)", "Transforms spatial luminance to frequency distribution via 2D DFT.", "Periodic spikes or cross-lines reveal cloned texture repetition."))
    with f2:
        st.image(homo_res, caption="Homomorphic Residual Δ_illum(x,y)", use_container_width=True, clamp=True)
        H(make_dip_card("Homomorphic Illumination Filter", "ln(f) = ln(i) + ln(r) -> HPF(D0) -> exp", "Separates low-freq lighting i(x,y) from high-freq reflectance r(x,y).", "Inconsistent illumination shading on spliced donor patches."))

with t4:
    SL("Compression Domain & Information Entropy — Chapter 8")
    ela_vis, _ = compute_ela(img_bgr, quality=ela_q)
    bag = compute_block_artifact_grid(img_bgr, block_size=8)
    entropy_map = compute_local_entropy(gray, radius=4)
    e1, e2, e3 = st.columns(3, gap="medium")
    with e1:
        st.image(ela_vis, caption=f"Error Level Analysis (Q={ela_q})", use_container_width=True)
        H(make_dip_card("Error Level Analysis (ELA)", "E(x,y) = |f(x,y) - f_Q90(x,y)| * α", "JPEG reaches error equilibrium on repeated saves; foreign patches do not.", "Bright anomalous highlights over spliced/foreign regions."))
    with e2:
        st.image(bag, caption="Block Artifact Grid BAG(x,y)", use_container_width=True, clamp=True)
        H(make_dip_card("Block Artifact Grid (BAG)", "Δ²_x = [1, -2, 1], 8×8 DCT boundaries", "Measures step discontinuity along JPEG 8×8 block edges.", "Grid disruption or missing grid steps along spliced margins."))
    with e3:
        st.image(entropy_map, caption="Local Shannon Entropy H(x,y)", use_container_width=True, clamp=True)
        H(make_dip_card("Shannon Rank Entropy", "H = -Σ p(i) * log2(p(i))", "Measures local information content and pixel randomness in bits.", "Uniform patches or altered noise profiles show low entropy."))

with t5:
    SL("Tamper Localization & Verdict — Chapters 9 & 10")
    cm_map, inliers, is_cm = detect_copy_move(img_bgr, n_features=1800, ratio_thresh=cm_ratio, min_inliers=12)
    spl_map, _ = detect_splice_anomalies(img_bgr, ela_quality=ela_q, homomorphic_d0=homo_d0)
    active_map = cm_map if (is_cm and np.max(cm_map) > 0.25) else spl_map
    clean_mask, _ = refine_tamper_mask(active_map, min_area=min_b)
    overlay = img_rgb.copy(); overlay[clean_mask == 255] = [255, 51, 102]
    blended = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)
    v1, v2, v3 = st.columns(3, gap="medium")
    with v1:
        if is_cm and inliers:
            st.image(draw_copy_move_vectors(img_rgb, inliers), caption=f"Cloned Vectors ({len(inliers)} inliers)", use_container_width=True)
        else:
            st.image(active_map, caption="Anomaly Heatmap A(x,y)", use_container_width=True, clamp=True)
    with v2:
        st.image(clean_mask, caption="Morphological Mask g(x,y)", use_container_width=True)
    with v3:
        st.image(blended, caption="Forensic Tamper Overlay", use_container_width=True)
    H("<br>")
    pct = (np.count_nonzero(clean_mask) / clean_mask.size) * 100
    method = f"Copy-Move Cloning ({len(inliers)} SIFT+RANSAC Inliers)" if is_cm else "Splice Anomaly (ELA + Homomorphic + BAG Fusion)"
    if pct > 0.5 or (is_cm and len(inliers) >= 12):
        H(f"""<div class="verdict-tampered"><div class="verdict-icon">⚠️</div><div>
          <p class="verdict-title">TAMPERING CONFIRMED — COMPLIANCE VIOLATION DETECTED</p>
          <p class="verdict-body">Flagged region: <strong style="color:var(--crimson)">{pct:.2f}%</strong> of total image area
          &nbsp;·&nbsp; Detection Algorithm: <strong>{method}</strong>
          &nbsp;·&nbsp; Mathematical Confidence: <strong style="color:var(--cyan-400)">HIGH</strong></p></div></div>""")
    else:
        H("""<div class="verdict-authentic"><div class="verdict-icon">✅</div><div>
          <p class="verdict-title">ASSET INTEGRITY CERTIFIED — AUTHENTIC</p>
          <p class="verdict-body">No statistically significant cloning vectors, illumination divergence, or JPEG grid misalignment detected.</p></div></div>""")
