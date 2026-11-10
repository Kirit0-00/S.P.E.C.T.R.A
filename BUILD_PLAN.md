# System Architecture & Build Plan: Classical DIP Image Forensics

## 1. Directory Tree Architecture

All modules strictly adhere to the $\le 200$ lines-per-file constraint. Responsibilities are strictly partitioned according to classical Digital Image Processing principles.

```text
Prac_project/
├── app.py                         # Streamlit Interactive Dashboard UI (orchestrator only, < 200 lines)
├── config.py                      # Global hyperparameters and algorithmic constants
├── MEMORY.md                      # Phased memory ledger, signatures, tensor shapes, and edge cases
├── SKILLS_REGISTRY.md             # Custom agentic skills and testing tool definitions
├── BUILD_PLAN.md                  # Master architectural blueprint and mathematical specifications
│
├── src/
│   ├── __init__.py
│   │
│   ├── spatial/                   # MODULE 1: Spatial Domain Preprocessing & Filtering
│   │   ├── __init__.py
│   │   ├── transforms.py          # Grayscale, logarithmic, power-law (gamma) transforms (< 150 lines)
│   │   ├── equalization.py        # Global Histogram Equalization & CLAHE (< 120 lines)
│   │   └── linear_filters.py      # High-pass (Laplacian, Sobel) & Low-pass (Gaussian, Bilateral) (< 180 lines)
│   │
│   ├── frequency/                 # MODULE 2: Frequency Domain Transforms
│   │   ├── __init__.py
│   │   ├── fft_core.py            # 2D FFT, 2D IFFT, centering/shifting, spectrum visualization (< 140 lines)
│   │   ├── frequency_filters.py   # Ideal, Butterworth, and Gaussian LPF / HPF (< 150 lines)
│   │   └── homomorphic.py         # Illumination-reflectance decoupling & homomorphic filtering (< 160 lines)
│   │
│   ├── compression/               # MODULE 3: Compression & Information Entropy
│   │   ├── __init__.py
│   │   ├── ela.py                 # Error Level Analysis (quantization difference matrix) (< 160 lines)
│   │   ├── block_grid.py          # JPEG 8x8 DCT grid extraction and misalignment detection (< 170 lines)
│   │   └── entropy.py             # Local Shannon entropy & variance mapping (< 140 lines)
│   │
│   ├── detection/                 # MODULE 4: Forensic Analyzers
│   │   ├── __init__.py
│   │   ├── copy_move.py           # Keypoint extraction (SIFT/ORB), g2NN matching, RANSAC (< 190 lines)
│   │   ├── copy_move_blocks.py    # Block-based DCT / variance matching for uniform areas (< 180 lines)
│   │   └── splice_detector.py     # ELA + homomorphic illumination anomaly fusion (< 180 lines)
│   │
│   └── morphology/                # MODULE 5: Morphological Post-Processing & Refinement
│       ├── __init__.py
│       ├── operators.py           # Erosion, Dilation, Opening, Closing with structuring elements (< 140 lines)
│       ├── reconstruction.py      # Morphological reconstruction, hole filling, top-hat/black-hat (< 160 lines)
│       └── segmentation.py        # Otsu/adaptive thresholding, CCA, thinning/skeletonization (< 170 lines)
│
└── tests/
    ├── __init__.py
    ├── synthetic_forger.py        # Deterministic in-memory forgery generator (< 200 lines)
    ├── test_spatial.py            # Unit tests for spatial transforms & filters (< 150 lines)
    ├── test_frequency.py          # Unit tests for FFT & homomorphic filtering (< 150 lines)
    ├── test_compression.py        # Unit tests for ELA & block artifacts (< 150 lines)
    ├── test_copy_move.py          # Unit tests for copy-move detection (< 150 lines)
    ├── test_splice.py             # Unit tests for splice detection (< 150 lines)
    └── test_morphology.py         # Unit tests for morphological mask cleanup (< 150 lines)
```

---

## 2. Mathematical Data Flow Matrix

Trace of input image $f(x, y)$ transforming into binary tamper mask $g(x, y)$:

```
           +-------------------------------------------------------------+
           | Input Image: f(x, y) \in R^{H x W x 3}, uint8 in [0, 255]   |
           +-------------------------------------------------------------+
                                      |
                     [Grayscale / Luminance Extraction]
                                      v
                 I(x, y) = 0.299 R + 0.587 G + 0.114 B
                                      |
             +------------------------+------------------------+
             |                                                 |
             v                                                 v
   [Copy-Move Pipeline]                              [Splice Detection Pipeline]
             |                                                 |
1. Local Variance Gating:                            1. Illumination Decoupling:
   sigma^2(x, y) = E[I^2] - (E[I])^2                    I(x, y) = i(x, y) * r(x, y)
             |                                                 |
2. Feature Detection:                                2. Homomorphic Filtering:
   K = {(p_i, d_i)} via SIFT/ORB                        z(x, y) = ln(I(x, y) + eps)
   where sigma^2(p_i) > tau_flat                        Z(u, v) = F{z(x, y)}
             |                                          S(u, v) = H(u, v) * Z(u, v)
3. Generalized 2NN Matching:                            s(x, y) = exp(Re{F^-1{S(u, v)}})
   dist(d_i, d_j1) / dist(d_i, d_j2) < rho              Illumination Residual:
   subject to ||p_i - p_j1||_2 > d_min                  Delta_illum(x, y) = |I(x, y) - s(x, y)|
             |                                                 |
4. Geometric Consistency:                            3. Error Level Analysis (ELA):
   RANSAC on affine transform:                          f_Q(x, y) = Dequantize(Quantize(f, Q))
   p'_i = H * p_i                                       E_raw(x, y) = ||f(x, y) - f_Q(x, y)||_2
   Inlier cluster support >= N_min                      E_norm(x, y) = scale(E_raw(x, y))
             |                                                 |
5. Clustered Heatmap:                                4. Block Artifact Grid (BAG):
   A_CM(x, y) = Hull(Inlier Pairs)                      B(x, y) = |2 * I(x, y) - I(x+1, y) - I(x-1, y)|
             |                                                 |
             |                                       5. Anomaly Map Fusion:
             |                                          A_splice(x, y) = alpha * E_norm
             |                                                         + beta * Delta_illum
             |                                                         + gamma * B
             +------------------------+------------------------+
                                      |
                                      v
                        [Morphological Refinement]
                                      |
                    1. Adaptive / Otsu Thresholding:
                       T = argmax_{t} sigma_B^2(t)
                       g_raw(x, y) = 1 if A(x, y) >= T else 0
                                      |
                    2. Morphological Cleaning:
                       Opening: g_1 = (g_raw (-) B_se) (+) B_se    (Noise suppression)
                       Closing: g_2 = (g_1 (+) B_se) (-) B_se    (Bridging gaps)
                                      |
                    3. Geodesic Reconstruction & Hole Filling:
                       g_3 = FillHoles(g_2)
                                      |
                    4. Connected Component Filtering:
                       Remove blobs where Area(C_k) < theta_min
                                      |
                    5. Morphological Boundary / Skeleton:
                       Edge(g) = g_3 - (g_3 (-) B_3x3)
                                      v
                 +--------------------------------------------+
                 | Final Output Mask: g(x, y) \in {0, 255}    |
                 +--------------------------------------------+
```

---

## 3. Algorithmic Strategy

### 3.1 Copy-Move Detection
1. **Variance Filtering:**
   Uniform background patches (sky, flat walls) trigger spurious descriptor matches. We compute local variance over a $(2k+1) \times (2k+1)$ window:
   $$\sigma^2(x, y) = \frac{1}{(2k+1)^2} \sum_{u, v} I(x+u, y+v)^2 - \left(\frac{1}{(2k+1)^2} \sum_{u, v} I(x+u, y+v)\right)^2$$
   Keypoints located in flat regions ($\sigma^2(x, y) < \tau_{\text{flat}}$) are discarded upfront.
2. **Keypoint Extraction & Multi-Match g2NN:**
   - Detect keypoints using SIFT (Scale-Invariant Feature Transform) with fallback to ORB.
   - Standard 2NN ratio test fails when an identical object is cloned multiple times. We employ **generalized 2NN (g2NN)**: keypoints are ranked by descriptor Euclidean distance $d_1 \le d_2 \le \dots \le d_k$, keeping all matches satisfying $d_i / d_{i+1} < \rho$ (typically $\rho = 0.65$) provided spatial displacement exceeds Euclidean threshold $d_{\text{spatial}} > \delta_{\text{min}}$.
3. **RANSAC Rigid / Affine Verification:**
   - Random Sample Consensus (RANSAC) fits an affine transformation matrix $\mathbf{H} \in \mathbb{R}^{2 \times 3}$:
     $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} \\ h_{21} & h_{22} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} h_{13} \\ h_{23} \end{bmatrix}$$
   - Only transformation models with inlier count $N_{\text{inliers}} \ge \Theta_{\text{support}}$ and non-trivial transformation (excluding identity/degenerate scales) are accepted as positive tamper clusters.

### 3.2 Splice Detection
1. **Error Level Analysis (ELA):**
   - Resave input $f(x, y)$ under standard JPEG compression with quality factor $Q_{\text{ref}} = 90$ or $95$, yielding $f_{Q}(x, y)$.
   - Compute channel-wise absolute error: $E(x, y) = |f(x, y) - f_{Q}(x, y)|$.
   - Authentic regions that have been repeatedly compressed converge to stable quantization states (low error), whereas spliced foreign patches (from uncompressed sources or different quality factors) exhibit elevated residual variance.
2. **Homomorphic Filtering for Illumination Inconsistency:**
   - Formulate image illumination-reflectance model: $I(x, y) = i(x, y) \cdot r(x, y)$, where $i(x, y)$ (incident light) is dominated by low spatial frequencies and $r(x, y)$ (reflectance) by high spatial frequencies.
   - Natural log decouples multiplicative components into additive terms: $\ln(I) = \ln(i) + \ln(r)$.
   - Apply 2D FFT: $\mathcal{F}\{\ln(I)\} = I(u, v)$.
   - Filter with modified high-emphasis Butterworth transfer function:
     $$H(u, v) = (\gamma_H - \gamma_L) \left[ 1 - \exp\left( -c \frac{D(u, v)^2}{D_0^2} \right) \right] + \gamma_L$$
     with $\gamma_L < 1$ (attenuating illumination) and $\gamma_H > 1$ (amplifying reflectance).
   - Invert log-Fourier transform: spliced components pasted from different lighting environments display distinct discontinuity profiles along boundary contours in the estimated illumination field $\hat{i}(x, y)$.
3. **Information Fusion & Morphological Segmentation:**
   - Combine ELA variance map, homomorphic illumination gradient, and 8x8 block-grid mismatch into composite anomaly response $A(x, y)$.
   - Binarize via Otsu's optimal global thresholding.
   - Clean with mathematical morphology: Opening ($B_{3 \times 3}$) removes noise spikes, Closing ($B_{7 \times 7}$) welds adjacent segments, Geodesic hole-filling seals interior voids, and Connected Component Analysis prunes regions smaller than minimum tamper threshold ($< 150$ px).

---

## 4. Testing & Verification Strategy (`tests/synthetic_forger.py`)

To ensure end-to-end validation without requiring external benchmark datasets during CI, `tests/synthetic_forger.py` generates in-memory deterministic forgeries with exact ground-truth masks:

1. **Synthetic Base Generators:**
   - Procedural textured scenes: Perlin/fractal noise, harmonic sinusoidal gratings, multi-channel naturalistic gradients.
2. **Copy-Move Synthesis Engine:**
   - Selects arbitrary geometric polygon $P \subset f(x, y)$.
   - Applies affine transforms: rotation $\theta \in [-180^\circ, 180^\circ]$, scale $s \in [0.5, 2.0]$, and translation vector $(\Delta x, \Delta y)$.
   - Blends target patch using:
     - Hard boundary pasting.
     - Gaussian boundary feathering ($k_{\text{blur}} \in \{3, 5, 7\}$).
   - Generates exact binary ground-truth mask $M_{\text{gt}}(x, y) = 1$ over the copied-to destination.
3. **Splice Synthesis Engine:**
   - Imports or synthesizes foreign donor patch $D(x, y)$ characterized by differing compression history ($Q_{\text{donor}} = 65$ vs $Q_{\text{base}} = 95$) or artificial illumination gradient $i_{\text{donor}}(x, y) = I_0 \exp(-\alpha (x^2 + y^2))$.
   - Inserts donor patch into host scene; exports paired ground truth.
4. **Validation Test Suite:**
   - Each detector must score $F_1 \ge 0.82$ on synthetic benchmarks with noise/compression levels up to $15\%$.
