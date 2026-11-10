# DIP Image Forensics: Active System Memory Ledger

## 1. Project Specifications
- **Objective:** Pure Classical Digital Image Processing (DIP) Image Forensics System (Copy-Move & Splice Detection).
- **Runtime:** Python 3.10+
- **Core Libraries:** `numpy`, `scipy`, `skimage` (scikit-image), `cv2` (OpenCV-Python), `matplotlib`, `streamlit`.
- **Architectural Constraints:**
  - Maximum 200 lines per `.py` source file (strictly verified: max file is 117 lines).
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
| `tests.synthetic_forger` | `generate_synthetic_base(H, W)` | `(H, W)` dimensions | `(H, W, 3)`, `uint8` synthetic scene | Completed & Tested |
| `tests.synthetic_forger` | `create_synthetic_copy_move(...)` | `(H, W, 3)`, patch & dest pos | `tuple(forged_img, gt_mask)` | Completed & Tested |
| `tests.synthetic_forger` | `create_synthetic_splice(...)` | `(H, W, 3)`, quality & patch | `tuple(forged_img, gt_mask)` | Completed & Tested |

---

## 3. Data Transformations & Tensor Shapes
- Color Image Input: $f(x, y) \in \mathbb{R}^{H \times W \times 3}$, dtype `uint8` (BGR or RGB) or `float64` in $[0, 1]$.
- Grayscale Luminance: $I(x, y) \in \mathbb{R}^{H \times W}$, dtype `float64` in $[0, 1]$ or `uint8` in $[0, 255]$.
- Frequency Domain Matrix: $F(u, v) \in \mathbb{C}^{H \times W}$, complex Fourier coefficients.
- Anomaly Heatmaps: $A(x, y) \in \mathbb{R}^{H \times W}$, dtype `float64` normalized to $[0.0, 1.0]$.
- Binary Tamper Mask: $g(x, y) \in \{0, 1\}^{H \times W}$, dtype `uint8` (`0` = authentic, `255` = forged).

---

## 4. Known Edge Cases & Handling Protocols
1. **Low Texture / Uniform Flat Regions:**
   - *Risk:* Keypoint starvation in SIFT/ORB, false positives in variance-based ELA.
   - *Protocol:* Handled via `compute_local_variance(img, ksize)`; flat regions where $\sigma^2(x, y) < \epsilon_{\text{flat}}$ are masked out.
2. **Lossless / Non-JPEG Inputs (PNG/TIFF/BMP):**
   - *Risk:* ELA baseline discrepancy because original image never suffered JPEG quantization.
   - *Protocol:* Handled by baseline re-encoding at $Q_{\text{ref}} = 90$ with dynamic normalization.
3. **High Dynamic Range & Extreme Illuminations:**
   - *Risk:* Logarithmic transforms blow up when $I(x, y) \to 0$ ($\ln(0)$ singularity).
   - *Protocol:* Handled via numerical epsilon offset: $\ln(I(x, y) + \epsilon)$ with $\epsilon = 10^{-5}$ in `src/frequency/homomorphic.py`.
4. **Resizing & Dimension Padding in 2D FFT:**
   - *Risk:* 2D FFT performance degraded by odd/prime dimensions.
   - *Protocol:* Handled via `scipy.fft.next_fast_len` optimal zero-padding with crop back to original shape in `compute_2d_ifft`.

---

## 5. Execution State
- **Current Phase:** PHASE 2 in progress (Modules 1 & 2 Complete: Spatial Preprocessing & Frequency Transforms Verified with 100% Passing Tests).
- **Compliance Status:** 0 Deep Learning imports, 100% DIP syllabus mapping, all files $\le 117$ lines.
