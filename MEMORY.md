# S.P.E.C.T.R.A.: Spatial Processing and Entropy Checks for Tamper Recognition Analysis
## Active System Memory Ledger

## 1. Project Specifications
- **Objective:** Pure Classical Digital Image Processing (DIP) Image Forensics System (Copy-Move & Splice Detection).
- **Runtime:** Python 3.10+
- **Core Libraries:** `numpy`, `scipy`, `skimage` (scikit-image), `cv2` (OpenCV-Python), `matplotlib`, `streamlit`.
- **Architectural Constraints:**
  - Maximum 200 lines per `.py` source file (strictly verified: all files <= 171 lines).
  - Zero Deep Learning (No Torch/TensorFlow/Keras/ONNX).
  - Explicit mathematical rigor mapping directly to classic DIP syllabus modules.

---

## 2. Function Signatures & Data Contracts

| Module | Function Signature | Expected Input Shape / Type | Output Shape / Type | Status |
| :--- | :--- | :--- | :--- | :--- |
| `src.spatial.transforms` | `to_grayscale(img)` | `(H, W, 3)` or `(H, W)`, `uint8` / `float` | `(H, W)`, `uint8` in `[0, 255]` | Completed & Tested |
| `src.spatial.transforms` | `to_float64(img)` | `(H, W)` or `(H, W, C)`, `numeric` | `(H, W)` or `(H, W, C)`, `float64` in `[0, 1]` | Completed & Tested |
| `src.spatial.transforms` | `normalize_to_uint8(img)` | `(H, W)` or `(H, W, C)`, `numeric` | `(H, W)` or `(H, W, C)`, `uint8` in `[0, 255]` | Completed & Tested |
| `src.spatial.transforms` | `log_transform(img, c)` | `(H, W)`, `uint8` / `float` | `(H, W)`, `float64` | Completed & Tested |
| `src.spatial.transforms` | `gamma_transform(img, gamma, c)` | `(H, W)`, `uint8` / `float` | `(H, W)`, `float64` in `[0, 1]` | Completed & Tested |
| `src.spatial.equalization` | `global_histogram_equalization(img)` | `(H, W)` or `(H, W, 3)`, `uint8` | `(H, W)` or `(H, W, 3)`, `uint8` | Completed & Tested |
| `src.spatial.equalization` | `apply_clahe(img, clip_limit, tile_grid_size)` | `(H, W)` or `(H, W, 3)`, `uint8` | `(H, W)` or `(H, W, 3)`, `uint8` | Completed & Tested |
| `src.spatial.equalization` | `compute_normalized_histogram(gray, bins)` | `(H, W)`, `uint8` | `(bins,)`, `float64` summing to 1.0 | Completed & Tested |
| `src.spatial.linear_filters` | `apply_gaussian_blur(img, ksize, sigma)` | `(H, W)` or `(H, W, C)`, `uint8` | `(H, W)` or `(H, W, C)`, `uint8` | Completed & Tested |
| `src.spatial.linear_filters` | `apply_bilateral_filter(img, d, sigmaColor, sigmaSpace)` | `(H, W, 3)` or `(H, W)`, `uint8` | `(H, W, 3)` or `(H, W)`, `uint8` | Completed & Tested |
| `src.spatial.linear_filters` | `compute_sobel_gradients(img, ksize)` | `(H, W)` or `(H, W, 3)`, `uint8` | `tuple(G_x, G_y, Mag)`, `float64` in `[0, 1]` | Completed & Tested |
| `src.spatial.linear_filters` | `apply_laplacian(img, ksize)` | `(H, W)` or `(H, W, 3)`, `uint8` | `(H, W)`, `float64` in `[0, 1]` | Completed & Tested |
| `src.spatial.linear_filters` | `apply_unsharp_mask(img, ksize, sigma, k)` | `(H, W)` or `(H, W, 3)`, `uint8` | `(H, W)` or `(H, W, 3)`, `uint8` in `[0, 255]` | Completed & Tested |
| `src.spatial.linear_filters` | `compute_local_variance(img, ksize)` | `(H, W)` or `(H, W, 3)`, `uint8` | `(H, W)`, `float64` | Completed & Tested |
| `src.frequency.fft_core` | `compute_2d_fft(img, pad_optimal)` | `(H, W)`, `uint8` / `float64` | `tuple(F_shifted, (H, W))`, `complex128` | Completed & Tested |
| `src.frequency.fft_core` | `compute_2d_ifft(f_shifted, target_shape)` | `(M, N)`, `complex128` | `(H, W)`, `float64` | Completed & Tested |
| `src.frequency.fft_core` | `get_magnitude_spectrum(f_shifted, log_scale)` | `(M, N)`, `complex128` | `(M, N)`, `float64` in `[0, 1]` | Completed & Tested |
| `src.frequency.fft_core` | `get_phase_spectrum(f_shifted)` | `(M, N)`, `complex128` | `(M, N)`, `float64` in `[-pi, pi]` | Completed & Tested |
| `src.frequency.frequency_filters` | `compute_distance_matrix(shape)` | `(M, N)` tuple | `(M, N)`, `float64` Euclidean distances | Completed & Tested |
| `src.frequency.frequency_filters` | `create_gaussian_filter(shape, d0, high_pass)` | `(M, N)`, `float`, `bool` | `(M, N)`, `float64` transfer function | Completed & Tested |
| `src.frequency.frequency_filters` | `create_butterworth_filter(shape, d0, order, high_pass)` | `(M, N)`, `float`, `int`, `bool` | `(M, N)`, `float64` transfer function | Completed & Tested |
| `src.frequency.frequency_filters` | `create_ideal_filter(shape, d0, high_pass)` | `(M, N)`, `float`, `bool` | `(M, N)`, `float64` transfer function | Completed & Tested |
| `src.frequency.frequency_filters` | `apply_frequency_filter(f_shifted, h_filter)` | `(M, N)` complex, `(M, N)` float | `(M, N)`, `complex128` filtered spectrum | Completed & Tested |
| `src.frequency.homomorphic` | `create_homomorphic_filter(shape, d0, gL, gH, c)` | `(M, N)`, `float` params | `(M, N)`, `float64` transfer function | Completed & Tested |
| `src.frequency.homomorphic` | `apply_homomorphic_filter(img, d0, gL, gH, c, eps)` | `(H, W)`, `uint8` / `float` | `tuple(enhanced, residual)`, `float64` | Completed & Tested |
| `src.compression.ela` | `compute_ela(img, quality, multiplier)` | `(H, W, 3)` or `(H, W)`, `uint8` | `tuple(visual uint8, raw float64)` | Completed & Tested |
| `src.compression.ela` | `compute_ela_variance(ela_raw, ksize)` | `(H, W)`, `float64` | `(H, W)`, `float64` in `[0.0, 1.0]` | Completed & Tested |
| `src.compression.block_grid` | `compute_block_artifact_grid(img, block_size)` | `(H, W)` or `(H, W, 3)`, `uint8` | `(H, W)`, `float64` in `[0.0, 1.0]` | Completed & Tested |
| `src.compression.block_grid` | `compute_grid_periodicity(img, block_size)` | `(H, W)`, `uint8` / `float64` | `dict(h_score, v_score)` in `[0.0, 1.0]` | Completed & Tested |
| `src.compression.entropy` | `compute_local_entropy(img, radius, selem)` | `(H, W)` or `(H, W, 3)`, `uint8` | `(H, W)`, `float64` in `[0.0, 1.0]` | Completed & Tested |
| `src.compression.entropy` | `compute_global_entropy(img)` | `(H, W)` or `(H, W, 3)`, `uint8` | `float` in `[0.0, 8.0]` bits | Completed & Tested |
| `src.detection.copy_move` | `extract_features_with_variance_gating(img, ...)` | `(H, W, 3)` or `(H, W)`, `uint8` | `tuple(keypoints, descriptors)` | Completed & Tested |
| `src.detection.copy_move` | `match_copy_move_features(...)` | `list(kps)`, `(N, D)` descriptors | `list(matched_pairs)` | Completed & Tested |
| `src.detection.copy_move` | `detect_copy_move(img, ...)` | `(H, W, 3)` or `(H, W)`, `uint8` | `tuple(heatmap, inlier_pairs, is_detected)` | Completed & Tested |
| `src.detection.splice_detector` | `detect_splice_anomalies(img, ...)` | `(H, W, 3)` or `(H, W)`, `uint8` | `tuple(fused_heatmap, cues_dict)` | Completed & Tested |
| `src.morphology.operators` | `get_structuring_element(shape, ksize)` | `str`, `int` | `(ksize, ksize)`, `uint8` binary kernel | Completed & Tested |
| `src.morphology.operators` | `apply_erosion(img, ksize, shape, iter)` | `(H, W)`, `uint8` | `(H, W)`, `uint8` | Completed & Tested |
| `src.morphology.operators` | `apply_dilation(img, ksize, shape, iter)` | `(H, W)`, `uint8` | `(H, W)`, `uint8` | Completed & Tested |
| `src.morphology.operators` | `apply_opening(img, ksize, shape)` | `(H, W)`, `uint8` | `(H, W)`, `uint8` | Completed & Tested |
| `src.morphology.operators` | `apply_closing(img, ksize, shape)` | `(H, W)`, `uint8` | `(H, W)`, `uint8` | Completed & Tested |
| `src.morphology.operators` | `apply_morphological_gradient(img, ksize)` | `(H, W)`, `uint8` | `(H, W)`, `uint8` | Completed & Tested |
| `src.morphology.reconstruction` | `fill_morphological_holes(binary_mask)` | `(H, W)`, `uint8` in `{0, 255}` | `(H, W)`, `uint8` in `{0, 255}` | Completed & Tested |
| `src.morphology.reconstruction` | `apply_tophat(img, ksize, shape)` | `(H, W)`, `uint8` | `(H, W)`, `uint8` | Completed & Tested |
| `src.morphology.reconstruction` | `apply_blackhat(img, ksize, shape)` | `(H, W)`, `uint8` | `(H, W)`, `uint8` | Completed & Tested |
| `src.morphology.reconstruction` | `morphological_reconstruction(marker, mask)` | `(H, W)`, `(H, W)` | `(H, W)`, `uint8` | Completed & Tested |
| `src.morphology.segmentation` | `apply_otsu_threshold(heatmap)` | `(H, W)`, `float64` in `[0.0, 1.0]` | `tuple(binary uint8, thresh float)` | Completed & Tested |
| `src.morphology.segmentation` | `filter_connected_components(binary, min_area)`| `(H, W)`, `uint8` in `{0, 255}` | `(H, W)`, `uint8` in `{0, 255}` | Completed & Tested |
| `src.morphology.segmentation` | `extract_morphological_boundary(binary)` | `(H, W)`, `uint8` in `{0, 255}` | `(H, W)`, `uint8` in `{0, 255}` | Completed & Tested |
| `src.morphology.segmentation` | `compute_skeleton(binary)` | `(H, W)`, `uint8` in `{0, 255}` | `(H, W)`, `uint8` in `{0, 255}` | Completed & Tested |
| `src.morphology.segmentation` | `refine_tamper_mask(heatmap, ...)` | `(H, W)`, `float64` in `[0.0, 1.0]` | `tuple(clean_mask, boundary)` in `{0, 255}` | Completed & Tested |
| `src.detection.copy_move_blocks`| `detect_copy_move_blocks(img, ...)` | `(H, W, 3)` or `(H, W)`, `uint8` | `tuple(heatmap float64, is_detected bool)` | Completed & Tested |
| `app.py` | Streamlit Dashboard Orchestrator | UI user interactions | Interactive Web Visualization | Completed |

---

## 3. Data Transformations & Tensor Shapes
- Color Image Input: $f(x, y) \in \mathbb{R}^{H \times W \times 3}$, dtype `uint8` (BGR or RGB) or `float64` in $[0, 1]$.
- Grayscale Luminance: $I(x, y) \in \mathbb{R}^{H \times W}$, dtype `float64` in $[0, 1]$ or `uint8` in $[0, 255]$.
- Frequency Domain Matrix: $F(u, v) \in \mathbb{C}^{H \times W}$, complex Fourier coefficients.
- Anomaly Heatmaps: $A(x, y) \in \mathbb{R}^{H \times W}$, dtype `float64` normalized to $[0.0, 1.0]$.
- Binary Tamper Mask: $g(x, y) \in \{0, 1\}^{H \times W}$, dtype `uint8` (`0` = authentic, `255` = forged).

---

## 4. Known Edge Cases & Handling Protocols
1. **Low Texture / Uniform Flat Regions:** Handled via `compute_local_variance(img, ksize)` gating ($\sigma^2(x, y) < \tau_{\text{flat}}$ excluded from descriptor matching).
2. **Lossless / Non-JPEG Inputs (PNG/TIFF/BMP):** Handled by baseline re-encoding at $Q_{\text{ref}} = 90$ with dynamic min-max normalization.
3. **High Dynamic Range & Extreme Illuminations:** Handled via numerical epsilon offset $\ln(I(x, y) + \epsilon)$ with $\epsilon = 10^{-5}$ in homomorphic filtering.
4. **Resizing & Dimension Padding in 2D FFT:** Handled via `scipy.fft.next_fast_len` optimal composite zero-padding with exact cropping in inverse transform.
5. **Small Isolated Noise Specks in Anomaly Heatmaps:** Eliminated via Otsu binarization + morphological opening ($B_{\text{ellipse}}$) + Connected Component Analysis minimum area threshold ($A \ge 150$ px).

---

## 5. Execution State
- **Current Phase:** PHASE 2 & 3 COMPLETE (All 5 Classical DIP Core Modules, Synthetic Testbeds, Unit Tests, and Streamlit Dashboard Implemented and 100% Passing).
- **Compliance Status:** 0 Deep Learning imports, 100% DIP syllabus mapping, all files $\le 171$ lines.
