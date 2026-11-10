# S.P.E.C.T.R.A. — DIP Concepts, Theory & Project Walkthrough
### Spatial Processing and Entropy Checks for Tamper Recognition Analysis

> **How to use this document:** Each section explains *what* the concept is in simple terms, *why* we need it for forensics, *the core math*, and then shows *exactly where and how* the project code uses it — with a short pseudo-code snippet tied to the actual file.

---

## Table of Contents

1. [The Big Picture — What Are We Detecting?](#1-the-big-picture)
2. [Module 1 — Spatial Domain (Preprocessing)](#2-module-1--spatial-domain)
3. [Module 2 — Frequency Domain & Homomorphic Filtering](#3-module-2--frequency-domain)
4. [Module 3 — Compression & ELA](#4-module-3--compression--ela)
5. [Module 4 — Detection (Copy-Move & Splice)](#5-module-4--detection)
6. [Module 5 — Morphological Mask Refinement](#6-module-5--morphological-refinement)
7. [End-to-End Flow Summary](#7-end-to-end-flow-summary)
8. [Viva Quick-Reference Table](#8-viva-quick-reference-table)

---

## 1. The Big Picture

### What is Image Forgery?

A digital image is just a 2D grid of numbers — $f(x, y)$ — where each cell holds a pixel intensity (0 = black, 255 = white for grayscale).

Two common types of forgery are:

| Forgery Type | What happens | Visual clue |
|---|---|---|
| **Copy-Move** | A region *within the same image* is copied and pasted elsewhere to hide something | Duplicate texture/pattern |
| **Splicing** | A patch is taken *from a different image* and pasted onto the host | Mismatched lighting, different compression noise |

### Why Classical DIP instead of AI?

Deep learning forensics is a black box — it cannot explain *why* it thinks an image is fake. Classical DIP uses explicit, explainable math:

- Every step is a known equation
- Intermediate results (heatmaps) are viewable and interpretable
- No training data needed — works on any image

---

## 2. Module 1 — Spatial Domain

> **Files:** [`src/spatial/transforms.py`](src/spatial/transforms.py), [`src/spatial/equalization.py`](src/spatial/equalization.py), [`src/spatial/linear_filters.py`](src/spatial/linear_filters.py)
> **DIP Chapter:** Gonzalez & Woods, Chapter 3

Spatial processing means operating directly on pixel values at their $(x, y)$ positions. Formally:
$$g(x, y) = T[f(x, y)]$$
where $T$ is some transformation applied to the pixel neighborhood.

---

### 2.1 Grayscale Conversion

**Simple idea:** Colour cameras have three channels (R, G, B). For forensic edge analysis, we work in a single luminance channel. The human eye is most sensitive to green, so we weight accordingly.

$$I(x, y) = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

**Where used in project:** Every processing module calls `to_grayscale()` first. This is the very first transformation applied to any input image.

```
# src/spatial/transforms.py → to_grayscale()

INPUT:  colour image f(x,y) of shape (H, W, 3)
  apply weighted sum: I = 0.299·R + 0.587·G + 0.114·B
OUTPUT: grayscale I of shape (H, W)        [used by ALL modules]
```

---

### 2.2 Log & Gamma Transforms (Gray-Level Transforms)

**Simple idea:** Some forgery seams hide in very dark or very bright areas. Log/gamma transforms stretch those hidden regions so they become visible.

| Transform | Formula | Effect |
|---|---|---|
| **Log** | $s = c \cdot \ln(1 + r)$ | Brightens dark areas, reveals shadow edges |
| **Gamma < 1** | $s = c \cdot r^\gamma$ | Expands dark region contrast |
| **Gamma > 1** | $s = c \cdot r^\gamma$ | Expands bright region contrast |

**Where used:** `apply_log_transform()` and `apply_gamma_correction()` in `transforms.py` are called during preprocessing before any filtering step.

```
# src/spatial/transforms.py → apply_log_transform()

INPUT:  image r ∈ [0, 255]
  r_norm = r / 255.0               # normalize to [0, 1]
  s      = ln(1 + r_norm)          # compress high, expand low
  s      = s / ln(2)               # normalize back to [0, 1]
OUTPUT: enhanced image s            [pre-processing step for display]
```

---

### 2.3 CLAHE — Contrast Enhancement

**Simple idea:** Regular Histogram Equalization (HE) makes the entire image's contrast even — but this over-brightens some regions and kills subtle seam detail.

**CLAHE (Contrast-Limited Adaptive Histogram Equalization)** does it *locally*:
1. Divide image into small tiles (e.g., 8×8 blocks)
2. Equalize histogram inside each tile, but *clip* it so loud areas don't dominate
3. Merge tiles smoothly using bilinear interpolation

This exposes feathering and blending artifacts along paste edges that global HE would miss.

**Math (histogram equalization inside each tile):**
$$s_k = (L-1) \sum_{j=0}^{k} \frac{n_j}{MN}$$
where $n_j$ = count of pixels with intensity $j$, $MN$ = tile area.

**Where used:** `apply_clahe()` in `equalization.py` is called in preprocessing to boost local contrast before SIFT keypoint detection and ELA analysis.

```
# src/spatial/equalization.py → apply_clahe()

INPUT:  grayscale image I
  split image into 8×8 tiles
  for each tile:
    compute local histogram (count of each intensity 0–255)
    clip histogram at limit β  (redistribute excess evenly)
    compute CDF of clipped histogram
    remap pixel intensities using CDF → s_k
  blend tile borders using bilinear interpolation
OUTPUT: contrast-enhanced image      [exposes feathering, clone edges]
```

---

### 2.4 Gaussian & Bilateral Filtering (Low-Pass)

**Simple idea:** Camera sensor noise is random and high-frequency. We need to smooth it out before detecting real anomalies — otherwise noise looks like a forgery signal.

**Gaussian filter** = weighted average of a pixel's neighborhood. The weight decreases with distance (bell curve):
$$G(x, y; \sigma) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$

**Problem:** Gaussian blurs sharp edges, which are exactly what we want to find along paste seams.

**Solution — Bilateral Filter:** Adds a second weight based on pixel *intensity difference*. If a neighbour is a very different colour, ignore it. This preserves edges while smoothing flat areas.

$$I_{\text{out}}(p) = \frac{\displaystyle\sum_{q} I(q) \cdot e^{-\|p-q\|^2/2\sigma_s^2} \cdot e^{-|I(p)-I(q)|^2/2\sigma_r^2}}{\text{normalization}}$$

**Where used:** `apply_gaussian_blur()` smooths before ELA; `apply_bilateral_filter()` smooths the final heatmap without destroying edge detail.

```
# src/spatial/linear_filters.py → apply_gaussian_blur()

INPUT:  image I, sigma
  build kernel W of size k×k using G(x, y; sigma)
  normalize W so sum(W) = 1
  output(x, y) = sum of W(i, j) · I(x+i, y+j) over k×k window
OUTPUT: smoothed image               [de-noises before ELA / SIFT]

# apply_bilateral_filter() — same idea but weight also by intensity diff
  weight(p, q) = spatial_weight(p, q) · intensity_weight(I(p), I(q))
  → edges are preserved; flat areas smoothed
```

---

### 2.5 Sobel & Laplacian (High-Pass / Edge Detection)

**Simple idea:** At a forgery boundary, pixel intensities change sharply. We detect these sharp changes using derivative filters.

**Sobel (1st derivative)** — detects edges in X and Y directions:
$$\mathbf{G}_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I \qquad \mathbf{G}_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I$$

Edge magnitude: $M(x,y) = \sqrt{G_x^2 + G_y^2}$

**Laplacian (2nd derivative)** — detects edges from all directions simultaneously. Flat areas → output = 0. Sharp boundaries → large value.
$$\mathbf{L} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

**Where used:** `compute_sobel_magnitude()` and `apply_laplacian_sharpening()` in `linear_filters.py` create edge maps used by the variance gating step and high-frequency anomaly heatmaps.

```
# src/spatial/linear_filters.py → compute_sobel_magnitude()

INPUT:  grayscale image I
  Gx = convolve(I, sobel_x_kernel_3x3)   # horizontal edges
  Gy = convolve(I, sobel_y_kernel_3x3)   # vertical edges
  M  = sqrt(Gx² + Gy²)
OUTPUT: edge magnitude map M             [high values = edges = paste seam candidate]
```

---

### 2.6 Local Variance Gating

**Simple idea:** SIFT tries to find "interesting" keypoints. Flat areas (sky, blank walls) produce useless descriptors — we compute how much variation exists in a small window and discard flat points.

$$\sigma^2(x,y) = \overline{I^2} - \bar{I}^2$$

**Where used:** `compute_local_variance()` in `linear_filters.py` is called inside `extract_features_with_variance_gating()` in `copy_move.py` to prune useless keypoints before SIFT matching.

```
# src/spatial/linear_filters.py → compute_local_variance()

INPUT:  grayscale image I, window ksize=7
  for each pixel (x, y) over a 7×7 window:
    mean  = average pixel value in window
    mean2 = average of (pixel value)² in window
    var(x, y) = mean2 - mean²
OUTPUT: variance map                     [keypoints where var < 10.0 are DISCARDED]
```

---

## 3. Module 2 — Frequency Domain

> **Files:** [`src/frequency/fft_core.py`](src/frequency/fft_core.py), [`src/frequency/frequency_filters.py`](src/frequency/frequency_filters.py), [`src/frequency/homomorphic.py`](src/frequency/homomorphic.py)
> **DIP Chapter:** Gonzalez & Woods, Chapter 4

Spatial domain shows *where* pixels are. Frequency domain shows *how fast* they change. Slow changes = low frequency (gradients, lighting). Fast changes = high frequency (edges, texture, noise).

The key mathematical tool is the **2D Discrete Fourier Transform (DFT)**:

$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)}$$

A fundamental property (Convolution Theorem):
$$f(x, y) * h(x, y) \xrightarrow{\text{FFT}} F(u,v) \cdot H(u,v)$$
Applying a filter in spatial domain (expensive convolution) = simple multiplication in frequency domain.

---

### 3.1 2D FFT — Computing the Spectrum

**Simple idea:** Transform the image into "how much of each frequency it contains." The center of the spectrum = slow/large patterns. The edges of the spectrum = fast/fine texture.

**Where used:** `compute_2d_fft()` in `fft_core.py` is called by both the frequency filters and the homomorphic filter as the entry point to the frequency domain.

```
# src/frequency/fft_core.py → compute_2d_fft()

INPUT:  image (H, W) or (H, W, 3)
  if colour: convert to grayscale
  pad image dimensions to next power-of-2  (faster FFT)
  F = fft2(image)              # complex 2D FFT
  F = fftshift(F)              # move DC (zero-freq) to center
OUTPUT: F (complex), padded_shape

  # To visualize spectrum:
  spectrum = log(1 + |F|)      # log scale makes dim parts visible
```

---

### 3.2 Frequency Domain Filters (Low-Pass & High-Pass)

**Simple idea:** Once in frequency domain, we can surgically keep or remove specific frequency bands by multiplying with a filter mask $H(u,v)$.

Let $D(u,v)$ = distance from center of spectrum:

| Filter | Formula | Removes |
|---|---|---|
| **Gaussian LP** | $H = e^{-D^2/2D_0^2}$ | High freqs (noise/texture) |
| **Gaussian HP** | $H = 1 - e^{-D^2/2D_0^2}$ | Low freqs (illumination gradient) |
| **Butterworth LP** (order $n$) | $H = \frac{1}{1+(D/D_0)^{2n}}$ | Smooth cutoff, no ringing |

**Where used:** `apply_lowpass_filter()` and `apply_highpass_filter()` in `frequency_filters.py`. Used to isolate frequency-domain anomalies before the splice score is computed.

```
# src/frequency/frequency_filters.py → apply_lowpass_filter()

INPUT:  image I, cutoff D0
  F = fft2(I)                        # go to frequency domain
  D = distance_from_center(F.shape)  # radial distance matrix
  H = exp(-D² / 2·D0²)              # Gaussian low-pass mask
  F_filtered = F · H                 # multiply: kills high freqs
  result = real(ifft2(F_filtered))   # back to spatial domain
OUTPUT: low-pass filtered image
```

---

### 3.3 Homomorphic Filtering — Catching Lighting Mismatches

**Simple idea:** A spliced image has two regions lit by different light sources. We need to *separate* the lighting component from the actual image content to detect this.

**The key insight:** Any image can be modelled as:
$$f(x, y) = \underbrace{i(x, y)}_{\text{illumination (slow → low-freq)}} \times \underbrace{r(x, y)}_{\text{reflectance (fast → high-freq)}}$$

We can't separate them with normal filters because they're *multiplied*. **Solution:** Take the log — multiplication becomes addition:

$$\ln(f) = \underbrace{\ln(i)}_{\text{low-freq component}} + \underbrace{\ln(r)}_{\text{high-freq component}}$$

Now standard frequency filtering can separate them!

**Complete pipeline (5 steps):**

```
Step 1: z    = ln(f + ε)              [log: × becomes +, ε avoids ln(0)]
Step 2: Z    = FFT(z)                 [go to frequency domain]
Step 3: S    = H(u,v) · Z            [high-emphasis filter applied]
Step 4: s    = IFFT(S)               [back to spatial domain]
Step 5: g    = exp(s) − ε            [undo the log]

Residual = |f − g|                   [bright = illumination mismatch = splice suspect]
```

The filter $H(u,v)$ is designed to *suppress* illumination (low freq) and *amplify* reflectance (high freq):
$$H(u, v) = (\gamma_H - \gamma_L)\!\left[1 - e^{-c \cdot D^2/D_0^2}\right] + \gamma_L \qquad (\gamma_L=0.5,\ \gamma_H=2.0)$$

**Where used:** `apply_homomorphic_filter()` in `homomorphic.py` returns `(enhanced_image, residual_map)`. The residual map feeds directly into the splice score with weight 0.3.

```
# src/frequency/homomorphic.py → apply_homomorphic_filter()

INPUT:  image (H, W, 3) or (H, W)
  gray = to_grayscale(image) / 255.0
  z    = log(gray + 1e-5)                # Step 1
  Z, _ = fft2(z, pad_optimal=True)      # Step 2
  H    = build_homo_filter(shape,        # Step 3
           gamma_L=0.5, gamma_H=2.0, D0=30)
  S    = Z · H
  s    = ifft2(S, target_shape)         # Step 4
  g    = exp(s) − 1e-5                  # Step 5
  residual = |gray − g| / max(|gray − g|)
OUTPUT: (enhanced_image, residual)       [residual = illumination anomaly map ∈ [0,1]]
```

---

## 4. Module 3 — Compression & ELA

> **Files:** [`src/compression/ela.py`](src/compression/ela.py), [`src/compression/block_grid.py`](src/compression/block_grid.py), [`src/compression/entropy.py`](src/compression/entropy.py)
> **DIP Chapter:** Gonzalez & Woods, Chapter 8

JPEG compression is lossy — it throws away information in a mathematically structured way. The signatures of this loss are exploited by our forensic tools.

---

### 4.1 JPEG Compression — Brief Overview

1. Divide image into 8×8 pixel blocks
2. Apply **DCT** (Discrete Cosine Transform) to each block:
   $$F_{\text{DCT}}(u, v) = \frac{1}{4} C(u)C(v) \sum_{x=0}^{7}\sum_{y=0}^{7} f(x,y) \cos\!\left[\frac{(2x+1)u\pi}{16}\right]\cos\!\left[\frac{(2y+1)v\pi}{16}\right]$$
3. **Quantize**: divide DCT coefficients by standard matrix $Q$ and round → information *thrown away* here
4. Huffman-encode the quantized integers

The quantization step is irreversible — and that's exactly what ELA exploits.

---

### 4.2 Error Level Analysis (ELA)

**Simple idea:** Take the suspect image and re-save it at a known JPEG quality (e.g., 90%). Compute the pixel-by-pixel difference.

- **Authentic region:** Already at JPEG equilibrium → re-saving barely changes it → **small error** (dark on ELA map)
- **Spliced region:** Different compression history → still has room to change → **large error** (bright on ELA map)

$$E(x,y) = \|f(x,y) - f_{Q=90}(x,y)\|$$

**Where used:** `compute_ela()` in `ela.py`. Returns `(ela_visual, ela_raw)`:
- `ela_visual` — amplified ×15 uint8 image for display in Streamlit
- `ela_raw` — normalized float map [0, 1] that feeds into the splice heatmap with weight **0.5** (highest weight)

```
# src/compression/ela.py → compute_ela()

INPUT:  image (H, W, 3) uint8
  encode image to in-memory JPEG buffer at quality=90
  decode buffer back  → resaved image
  diff = |original − resaved|  per channel
  ela_raw    = mean(diff, axis=channels) / 255   # anomaly float map ∈ [0,1]
  ela_visual = clip(diff × 15, 0, 255)           # amplified for display
OUTPUT: (ela_visual uint8, ela_raw float64)
        [ela_raw → splice_detector.py as E_norm]
```

---

### 4.3 Block Artifact Grid (BAG) Detection

**Simple idea:** JPEG encodes in 8×8 blocks, creating subtle steps at pixel coordinates that are multiples of 8 (positions 0, 8, 16, 24, ...). When a foreign patch is pasted, its own 8-pixel grid is usually *misaligned* with the host grid. We detect this misalignment.

We measure 2nd-order step discontinuities (finite difference):
$$D_h(x, y) = |2I(x,y) - I(x+1,y) - I(x-1,y)|$$
$$D_v(x, y) = |2I(x,y) - I(x,y+1) - I(x,y-1)|$$

**Where used:** `compute_block_artifact_grid()` in `block_grid.py`. Its output contributes to the splice score with weight **0.2**.

```
# src/compression/block_grid.py → compute_block_artifact_grid()

INPUT:  grayscale image I
  Dh = |2·I(x,y) - I(x+1,y) - I(x−1,y)|   # horizontal step
  Dv = |2·I(x,y) - I(x,y+1) - I(x,y−1)|   # vertical step
  BAG = (Dh + Dv) / 2
  Enhance by applying 8-pixel-periodic mask (highlight every 8th line)
OUTPUT: BAG map                             [bright = misaligned 8-px boundary = splice]
        [BAG → splice_detector.py as BAG term]
```

---

### 4.4 Shannon Entropy Map

**Simple idea:** Real camera noise is *random* → high entropy (many different values in a window). A region that was copy-pasted or generated is more *uniform* → lower entropy.

$$H(x,y) = -\sum_{k=0}^{255} p(r_k) \log_2 p(r_k) \qquad [\text{bits}]$$

Natural scene ≈ 7–8 bits. Heavily compressed/fake patch ≈ 4–5 bits.

**Where used:** `compute_local_entropy()` in `entropy.py`. Used as a supplementary confidence feature for the splice detector.

```
# src/compression/entropy.py → compute_local_entropy()

INPUT:  grayscale image I (must be uint8)
  use skimage.filters.rank.entropy(I, disk(5)):
    for each pixel, look at disk window of radius 5
    compute histogram of intensities
    H = −sum(p · log2(p))   for each intensity p
OUTPUT: entropy map (H, W)      [low entropy ≈ suspicious / forged region]
```

---

## 5. Module 4 — Detection

> **Files:** [`src/detection/copy_move.py`](src/detection/copy_move.py), [`src/detection/splice_detector.py`](src/detection/splice_detector.py)
> **DIP Chapter:** Gonzalez & Woods, Chapters 10 & 11

---

### 5.1 Copy-Move Detection (SIFT + g2NN + RANSAC)

**Simple idea:** If a region was cloned within the same image, there will be two or more patches of pixels that look almost identical, just at different locations. We detect these by:

1. Finding "interesting" points (keypoints) with unique 128-dim local descriptors (SIFT)
2. Finding keypoints that match each other (similar descriptor) but are far apart spatially
3. Verifying the matches make geometric sense (RANSAC)

#### Step A — Feature Extraction with Variance Gating

SIFT finds keypoints and builds a **128-dimensional descriptor** for each — a fingerprint of the local patch. Before running SIFT we filter out flat/boring regions using local variance (Section 2.6) to avoid spurious matches.

#### Step B — g2NN Matching (Generalized 2-Nearest Neighbours)

For each keypoint, find its nearest match in the **same image**. Accept the match only if:
1. The match distance is significantly smaller than the 2nd-best match: $\frac{d_1}{d_2} < 0.65$
2. The two matched points are at least 20 pixels apart (not the same point)

This "ratio test" filters out ambiguous matches — if two descriptors look equally similar to a third, neither is reliable.

#### Step C — RANSAC Geometric Verification

Fits an **affine transform** $\mathbf{H}$ to the matched pairs. Matches that fit the model are **inliers** (real clones); random noise matches are **outliers**. Minimum 4 inliers = copy-move confirmed.

$$\mathbf{p}' = \begin{bmatrix} h_{11} & h_{12} \\ h_{21} & h_{22} \end{bmatrix} \mathbf{p} + \begin{bmatrix} h_{13} \\ h_{23} \end{bmatrix}$$

**Where used:** `detect_copy_move()` in `copy_move.py` returns `(heatmap, inlier_pairs, is_detected)`.

```
# src/detection/copy_move.py → detect_copy_move()

INPUT:  image (H, W, 3)
  gray  = to_grayscale(image)
  var   = compute_local_variance(gray, ksize=7)

  # Step A: Extract SIFT features
  sift       = cv2.SIFT_create(nfeatures=1500)
  kps, descs = sift.detectAndCompute(gray, mask=None)
  kps        = [kp for kp in kps if var[kp.y, kp.x] >= 10.0]  # variance gate

  # Step B: Self-matching with g2NN ratio test
  bf      = BFMatcher(NORM_L2)
  matches = bf.knnMatch(descs, descs, k=4)       # k=4 to skip self (dist≈0)
  good    = [(kp1, kp2) for m1, m2 in matches
              if m1.dist / m2.dist < 0.65          # ratio test
              and spatial_dist(kp1, kp2) > 20]     # displacement check

  # Step C: RANSAC affine verification
  H, inliers = cv2.estimateAffine2D(pts1, pts2, method=RANSAC, threshold=5.0)
  if count(inliers) >= 4:
    paint Gaussian blobs on heatmap at inlier locations
    return heatmap, inlier_pairs, is_tampered=True

OUTPUT: (anomaly_heatmap ∈ [0,1], inlier_point_pairs, bool)
```

---

### 5.2 Splice Detection (Multi-Cue Fusion)

**Simple idea:** No single test is perfect. We combine three independent forensic signals into one final heatmap using a *weighted sum*.

$$A(x,y) = \underbrace{0.5 \cdot E_{\text{norm}}(x,y)}_{\text{ELA}} + \underbrace{0.3 \cdot \Delta_{\text{illum}}(x,y)}_{\text{Homomorphic residual}} + \underbrace{0.2 \cdot \text{BAG}(x,y)}_{\text{Block grid}}$$

| Signal | Source Module | Detects | Weight |
|---|---|---|---|
| ELA ($E_{\text{norm}}$) | `ela.py` | Different compression history | 0.5 |
| Illumination residual ($\Delta_{\text{illum}}$) | `homomorphic.py` | Mismatched lighting | 0.3 |
| BAG score | `block_grid.py` | Misaligned 8×8 JPEG blocks | 0.2 |

**Where used:** `run_splice_detection()` in `splice_detector.py` returns `(heatmap, is_tampered)`.

```
# src/detection/splice_detector.py → run_splice_detection()

INPUT:  image (H, W, 3)
  _, ela_raw  = compute_ela(image, quality=90)         # ELA float map [0,1]
  _, homo_res = apply_homomorphic_filter(image)        # illumination residual [0,1]
  bag         = compute_block_artifact_grid(gray)      # block grid anomaly [0,1]

  # Weighted fusion (weights sum to 1.0)
  A(x,y) = 0.5 · ela_raw + 0.3 · homo_res + 0.2 · bag

  # Tamper decision
  tamper_ratio = mean(A[A > threshold])
  is_tampered  = tamper_ratio > 0.005

OUTPUT: (heatmap A ∈ [0,1], is_tampered bool)
```

---

## 6. Module 5 — Morphological Refinement

> **Files:** [`src/morphology/operators.py`](src/morphology/operators.py), [`src/morphology/reconstruction.py`](src/morphology/reconstruction.py), [`src/morphology/segmentation.py`](src/morphology/segmentation.py)
> **DIP Chapter:** Gonzalez & Woods, Chapter 9

The raw anomaly heatmap $A(x,y)$ from Module 4 is a noisy, blurry float image. Morphology converts it into a clean, solid binary mask — white = forged region, black = authentic.

---

### 6.1 Thresholding with Otsu's Method

**Simple idea:** We need to convert the grayscale heatmap into a binary (black/white) mask. But what threshold should we use? **Otsu's method** automatically finds the best one by maximizing the contrast between the two resulting groups.

**How it works:**
1. Try every possible threshold $t$ from 0 to 255
2. For each $t$, compute the variance *between* background and foreground groups
3. Pick the $t$ that maximizes separation

$$t^* = \arg\max_t \; \omega_0(t) \cdot \omega_1(t) \cdot [\mu_0(t) - \mu_1(t)]^2$$

**Where used:** `otsu_threshold()` in `segmentation.py` — the first step in converting the heatmap to a binary mask.

```
# src/morphology/segmentation.py → otsu_threshold()

INPUT:  heatmap A ∈ [0.0, 1.0]
  heatmap_uint8 = (A × 255).astype(uint8)
  threshold, binary = cv2.threshold(
      heatmap_uint8,
      0,                    # 0 = auto, Otsu computes best t*
      255,
      THRESH_BINARY + THRESH_OTSU
  )
OUTPUT: binary_mask         [0 = authentic, 255 = suspicious]
```

---

### 6.2 Erosion & Dilation

Think of the binary mask as a **set** of "on" pixels (white = forged region). A structuring element (SE) is a small shape (e.g., 5×5 disk) used as a probe.

| Operation | What it does | Formula | Forensic Use |
|---|---|---|---|
| **Erosion** | Shrinks white regions | Keep pixel only if entire SE fits inside white area | Removes tiny noise dots |
| **Dilation** | Grows white regions | Add pixel if SE overlaps any white pixel | Bridges small gaps in a detected region |

$$\text{Erosion:} \quad A \ominus B = \{z \mid (B)_z \subseteq A\}$$
$$\text{Dilation:} \quad A \oplus B = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}$$

**Where used:** `apply_erosion()` and `apply_dilation()` in `operators.py`. Building blocks for Opening and Closing.

```
# src/morphology/operators.py → apply_erosion() / apply_dilation()

INPUT:  binary_mask, ksize=5, shape="ellipse"
  kernel = cv2.getStructuringElement(MORPH_ELLIPSE, (5, 5))

  # Erosion: a pixel stays white only if ALL kernel pixels are white
  eroded  = cv2.erode(binary_mask, kernel, iterations=1)

  # Dilation: a pixel turns white if ANY kernel pixel is white
  dilated = cv2.dilate(binary_mask, kernel, iterations=1)
```

---

### 6.3 Opening & Closing

These are combinations of erosion and dilation applied in sequence:

**Opening** = Erode → then Dilate:
$$A \circ B = (A \ominus B) \oplus B$$
→ **Removes small isolated noise blobs** (too small to survive erosion). Shape of large genuine regions is preserved.

**Closing** = Dilate → then Erode:
$$A \bullet B = (A \oplus B) \ominus B$$
→ **Fills small holes** inside detected regions. Bridges narrow gaps between parts of the same forged region.

**Applied in order:** Opening first (clean noise) → Closing second (seal gaps).

**Where used:** `apply_opening()` and `apply_closing()` in `operators.py`.

```
# src/morphology/operators.py → apply_opening() / apply_closing()

INPUT:  binary_mask after Otsu
  kernel = cv2.getStructuringElement(MORPH_ELLIPSE, (5, 5))

  # Step 1 — Opening: kills noise first
  opened = cv2.morphologyEx(binary_mask, MORPH_OPEN,  kernel)

  # Step 2 — Closing: seals gaps in real regions
  closed = cv2.morphologyEx(opened,      MORPH_CLOSE, kernel)

OUTPUT: cleaned binary mask
```

---

### 6.4 Hole Filling (Morphological Reconstruction)

**Simple idea:** After opening and closing, some forged regions may still have small holes *inside* them (pixels that look authentic but are completely surrounded by forged pixels). We fill these using a flood-fill approach:

$$D_G^{(1)}(F) = (F \oplus B) \cap G$$
Repeat until convergence (no more pixels change).

**Where used:** `fill_binary_holes()` in `reconstruction.py`.

```
# src/morphology/reconstruction.py → fill_binary_holes()

INPUT:  closed binary mask
  inverted    = 255 − closed_mask          # holes become white blobs
  flood_filled = cv2.floodFill(            # fill from all border pixels
                   inverted, seed=(0,0), newVal=0)
  filled = 255 − flood_filled             # flip back
OUTPUT: mask with interior holes filled    [solid forged region blobs]
```

---

### 6.5 Connected Component Analysis (CCA)

**Simple idea:** After morphological cleanup, tiny isolated specks of "white" remain — these are residual noise. CCA labels each connected white blob separately, then we eliminate any blob that is too small to be a real forged region (< 150 pixels).

**Where used:** `filter_by_area()` in `segmentation.py` — the final cleanup step.

```
# src/morphology/segmentation.py → filter_by_area()

INPUT:  filled binary mask
  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
  output = zeros_like(mask)
  for each label k (skip k=0 = background):
    area = stats[k, STAT_AREA]
    if area >= 150:                # minimum blob size threshold
      output[labels == k] = 255   # keep it
    # else: discard (too small = noise artifact)
OUTPUT: final clean tamper mask g(x,y) ∈ {0, 255}
```

---

### 6.6 Boundary Extraction

**Simple idea:** To draw a clean 1-pixel outline around the detected forged region (for overlay on the original image), we subtract an eroded version from the mask:

$$\beta(g) = g - (g \ominus B_{3\times3})$$

**Where used:** `extract_boundary()` in `segmentation.py`. The boundary pixels are drawn as the red/cyan overlay shown in the Streamlit UI.

```
# src/morphology/segmentation.py → extract_boundary()

INPUT:  final tamper mask g
  kernel_3x3 = ones((3,3))
  eroded      = cv2.erode(g, kernel_3x3, iterations=1)
  boundary    = g − eroded        # only 1-pixel outer ring survives
OUTPUT: boundary mask              [used for overlay visualization in app.py]
```

---

## 7. End-to-End Flow Summary

Here is the complete journey of a suspicious image through S.P.E.C.T.R.A.:

```
 Input Image f(x, y)  [H × W × 3, uint8]
        │
        ▼
 ┌──────────────────────────────────────────────────┐
 │  MODULE 1: SPATIAL PREPROCESSING                │
 │  to_grayscale()      → I(x,y)  [luminance]      │
 │  apply_clahe()       → enhanced contrast         │
 │  apply_gaussian()    → noise suppression         │
 │  compute_sobel()     → edge map                  │
 │  compute_local_var() → variance map (for SIFT)   │
 └───────────────────┬──────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 ┌─────────────────┐   ┌────────────────────────────┐
 │  MODULE 2: FFT  │   │  MODULE 3: COMPRESSION     │
 │  compute_2d_fft │   │  compute_ela()             │
 │  → Z(u,v)       │   │  → ela_raw [0,1]           │
 │  apply_homo_    │   │  compute_block_artifact()  │
 │  filter()       │   │  → BAG [0,1]               │
 │  → residual     │   │  compute_local_entropy()   │
 │    [0,1]        │   │  → entropy map             │
 └────────┬────────┘   └───────────┬────────────────┘
          │                        │
          └──────────┬─────────────┘
                     ▼
 ┌─────────────────────────────────────────────────────┐
 │  MODULE 4: DETECTION                                │
 │                                                     │
 │  ── Copy-Move path ──────────────────────────────  │
 │  SIFT keypoints → variance gate → g2NN match       │
 │  → RANSAC affine → inlier heatmap A_CM(x,y)        │
 │                                                     │
 │  ── Splice path ─────────────────────────────────  │
 │  A(x,y) = 0.5·ELA + 0.3·Homo + 0.2·BAG            │
 │  → splice heatmap A_SP(x,y)                        │
 └────────────────────┬────────────────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────────────┐
 │  MODULE 5: MORPHOLOGICAL MASK REFINEMENT         │
 │  otsu_threshold()    → binary g_raw(x,y)         │
 │  apply_opening()     → remove salt-pepper noise  │
 │  apply_closing()     → fill interior gaps        │
 │  fill_binary_holes() → solid blobs               │
 │  filter_by_area()    → remove blobs < 150px      │
 │  extract_boundary()  → 1-px overlay line         │
 └────────────────────┬─────────────────────────────┘
                      │
                      ▼
         Final Tamper Mask g(x, y) ∈ {0, 255}
         + Heatmap overlay on original image
         + Verdict: AUTHENTIC / TAMPERED
```

---

## 8. Viva Quick-Reference Table

| Question | Simple Answer | Key Formula / Code Location |
|---|---|---|
| **What is a spatial domain transform?** | Operating on pixel values directly in their (x,y) positions | $g(x,y) = T[f(x,y)]$ — `transforms.py` |
| **Why CLAHE over global Histogram Equalization?** | Global HE treats the whole image uniformly and kills local seam details. CLAHE works tile-by-tile and clips the histogram to prevent noise amplification | Clip at $\beta$, bilinear-blend tile borders — `equalization.py` |
| **What does the FFT spectrum show?** | DC center = overall brightness; moving outward = finer detail. Bright off-center spots = repeating patterns (copy-move signature) | $F(u,v) = \sum_{x,y} f e^{-j2\pi(\ldots)}$ — `fft_core.py` |
| **Why Homomorphic Filtering?** | Illumination $i$ and reflectance $r$ are multiplied: $f = i \cdot r$. Normal filters can't separate them. Log converts × to +, then FFT can filter them separately | $\ln(f) = \ln(i) + \ln(r)$ — `homomorphic.py` |
| **How does ELA detect splices?** | Authentic JPEG is at compression equilibrium — re-saving barely changes it. A pasted-in region from a different source still changes — its error residual is high | $E = \|f - f_{Q=90}\|$ — `ela.py` |
| **What is RANSAC for?** | SIFT matching produces many wrong matches. RANSAC fits an affine model and keeps only the matches that agree geometrically (inliers) | $\mathbf{p}' = H\mathbf{p}$, keep if reprojection error < 5 px — `copy_move.py` |
| **Why Opening before Closing?** | Opening first removes tiny noise spikes (they don't survive erosion). Closing then heals real gaps — without accidentally welding noise back in | Open: $(A\ominus B)\oplus B$, Close: $(A\oplus B)\ominus B$ — `operators.py` |
| **Why discard small components (CCA)?** | Real forged regions are at least a few hundred pixels. Tiny specks after morphology are residual noise — CCA + area filter removes them | Keep if Area$(C_k) \geq 150$ px — `segmentation.py` |
| **Why not use Deep Learning?** | DL is a black box — can't explain why it flags an image. Classical DIP uses explicit, verifiable equations at every single step | Zero `torch`/`tensorflow` imports — verified by CI |
| **What does entropy tell us forensically?** | Real sensor noise is random = high entropy ≈ 7–8 bits. Copy-pasted or generated uniform regions have lower entropy ≈ 4–5 bits | $H = -\sum p \log_2 p$ — `entropy.py` |
| **Why bilateral filter instead of Gaussian?** | Gaussian blurs edges too — including the very paste seams we want to find. Bilateral preserves sharp transitions by weighting on intensity similarity too | Two-kernel weight: spatial + intensity — `linear_filters.py` |
| **What does the splice score formula mean?** | Each of the three signals (ELA, Homomorphic, BAG) is imperfect alone. Weighted fusion reduces false positives from any single signal | $A = 0.5\cdot E + 0.3\cdot R + 0.2\cdot B$ — `splice_detector.py` |
