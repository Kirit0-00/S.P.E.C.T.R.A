# S.P.E.C.T.R.A. — Theoretical Foundations & Mathematical Manual
### Spatial Processing and Entropy Checks for Tamper Recognition Analysis
**A Comprehensive Classical Digital Image Processing (DIP) Pedagogical Reference**  
*Mapped directly to standard curricula and Gonzalez & Woods: "Digital Image Processing" (3rd/4th Edition)*

---

## Table of Contents
1. [Executive Summary & Forensic Philosophy](#1-executive-summary--forensic-philosophy)
2. [Module 1: Spatial Domain Preprocessing & Filtering (Chapter 3)](#2-module-1-spatial-domain-preprocessing--filtering-chapter-3)
   - 2.1 Luminance Extraction & Gray-Level Transformations
   - 2.2 Contrast Enhancement: Global Histogram Equalization & CLAHE
   - 2.3 Spatial Linear Operators: Gaussian & Bilateral Filtering
   - 2.4 High-Pass Spatial Edge Derivatives: Sobel Gradients & Laplacian
   - 2.5 Local Statistical Variance Gating
3. [Module 2: Frequency Domain Analysis & Homomorphic Filtering (Chapter 4)](#3-module-2-frequency-domain-analysis--homomorphic-filtering-chapter-4)
   - 3.1 2D Discrete Fourier Transform (DFT) & Centering Property
   - 3.2 Ideal, Gaussian, and Butterworth Frequency Domain Transfer Functions
   - 3.3 The Illumination-Reflectance Model & Homomorphic Filtering
4. [Module 3: Compression Redundancies & Information Entropy (Chapter 8)](#4-module-3-compression-redundancies--information-entropy-chapter-8)
   - 4.1 JPEG Compression Pipeline & Quantization Matrices
   - 4.2 Error Level Analysis (ELA) Physics & Equilibrium
   - 4.3 8×8 Block Artifact Grid (BAG) Step-Discontinuity Detection
   - 4.4 Local Shannon Information Entropy Mapping
5. [Module 4: Forensic Detection Orchestration (Chapters 10 & 11)](#5-module-4-forensic-detection-orchestration-chapters-10--11)
   - 5.1 Copy-Move Detection: Keypoints, g2NN Matching & RANSAC
   - 5.2 Splice Detection: Multi-Cue Convex Combination
6. [Module 5: Mathematical Morphology & Mask Refinement (Chapters 9 & 10)](#6-module-5-mathematical-morphology--mask-refinement-chapters-9--10)
   - 6.1 Set-Theoretic Operators: Erosion, Dilation, Opening, Closing
   - 6.2 Geodesic Hole-Filling & Morphological Reconstruction
   - 6.3 Otsu’s Optimal Between-Class Variance Thresholding
   - 6.4 Connected Component Analysis (CCA) & Skeleton Boundary Extraction
7. [End-to-End Mathematical Transformation Matrix](#7-end-to-end-mathematical-transformation-matrix)
8. [Conclusion & Viva / Examination Quick Reference](#8-conclusion--viva--examination-quick-reference)

---

## 1. Executive Summary & Forensic Philosophy

Digital images are discrete 2D spatial intensity functions $f(x, y)$. When an adversary performs an image forgery—either by cloning an element within the same frame (**Copy-Move**) or pasting foreign pixels from a donor source (**Splicing**)—they disrupt the natural physical invariants established during camera sensor capture.

Deep learning forensics models often operate as opaque "black boxes" prone to dataset bias and adversarial vulnerabilities. **S.P.E.C.T.R.A.** enforces a **100% Classical DIP approach**:
- Every step is an explicit mathematical operation on spatial, frequency, compression, or topological domains.
- Intermediate representations are directly viewable and mathematically quantifiable.
- Zero neural network weights, zero cloud dependencies.

---

## 2. Module 1: Spatial Domain Preprocessing & Filtering (Chapter 3)

Spatial processing operates directly on image coordinates $(x, y)$ via neighborhood transformations:
$$g(x, y) = T[f(x, y)]$$

### 2.1 Luminance Extraction & Gray-Level Transformations

#### Grayscale Conversion (ITU-R BT.601)
Color image $f(x, y) = [R(x, y), G(x, y), B(x, y)]^T$ is converted to scalar luminance $I(x, y)$ using human psychophysical perceptual weighting:
$$I(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$
*Forensic Motivation:* Human photoreceptors (cones) are predominantly sensitive to green wavelengths. Decoupling luminance eliminates chromatic distraction while preserving edge and gradient fidelity.

#### Logarithmic Dynamic Range Compression
$$s = c \cdot \ln(1 + r), \quad c = \frac{1}{\ln(1 + \max(r))}$$
*Forensic Motivation:* Compresses wide dynamic ranges to elevate low-intensity detail in dark underexposed shadow regions where tampering seams often lurk.

#### Power-Law (Gamma) Transformation
$$s = c \cdot r^\gamma$$
- $\gamma < 1$: Expands dark intensity levels (reveals hidden edge noise).
- $\gamma > 1$: Compresses shadows, expands highlights.

---

### 2.2 Contrast Enhancement: Global Histogram Equalization & CLAHE

#### Global Histogram Equalization (GHE)
Given a discrete grayscale image with intensity levels $r_k \in [0, L-1]$:
1. Probability of occurrence of intensity level $r_k$:
   $$p_r(r_k) = \frac{n_k}{MN}, \quad k = 0, 1, \dots, L-1$$
2. Cumulative Distribution Function (CDF) transformation:
   $$s_k = T(r_k) = (L - 1) \sum_{j=0}^k p_r(r_j)$$
*Forensic Limitation:* GHE is global; high-contrast global shifts over-saturate local micro-textures and suppress boundary seams.

#### Contrast-Limited Adaptive Histogram Equalization (CLAHE)
1. **Grid Partitioning:** The spatial domain is divided into $M \times N$ contextual sub-blocks (typically $8 \times 8$).
2. **Local Histogram Clipping:** To prevent over-amplifying noise in homogeneous areas, local histograms are clipped at threshold $\beta$:
   $$\text{Clip Limit } \beta = \frac{N_{\text{pixels}}}{N_{\text{bins}}} \left( 1 + \frac{\alpha}{100} (S_{\text{max}} - 1) \right)$$
   The clipped excess pixels are redistributed uniformly across all bins.
3. **Bilinear Interpolation:** Eliminates artificial boundary block lines between adjacent contextual tiles:
   $$I(x, y) = (1 - \Delta x)(1 - \Delta y) I_{A} + \Delta x(1 - \Delta y) I_{B} + (1 - \Delta x)\Delta y I_{C} + \Delta x \Delta y I_{D}$$
*Forensic Utility:* Exposes feathering boundaries and subtle clone edges without amplifying background sensor noise.

---

### 2.3 Spatial Linear Operators: Gaussian & Bilateral Filtering

#### 2D Gaussian Low-Pass Kernel
Linear spatial smoothing attenuates high-frequency noise via discrete convolution:
$$G(x, y; \sigma) = \frac{1}{2\pi \sigma^2} \exp\left( -\frac{x^2 + y^2}{2\sigma^2} \right)$$
Discrete kernel $W_{k \times k}$ is formed by sampling $G(x, y)$ and normalizing: $\sum_{u, v} W(u, v) = 1$.

#### Edge-Preserving Bilateral Filter
Standard Gaussian smoothing blurs tamper boundaries. The Bilateral Filter preserves sharp structural seams by combining domain (spatial) and range (radiometric) Gaussian kernels:
$$I^{\text{filtered}}(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \cdot g_s(\|p - q\|) \cdot g_r(|I(p) - I(q)|)$$
Where normalization factor:
$$W_p = \sum_{q \in \Omega} \exp\left(-\frac{\|p - q\|^2}{2\sigma_s^2}\right) \exp\left(-\frac{|I(p) - I(q)|^2}{2\sigma_r^2}\right)$$
*Forensic Utility:* Smooths natural sensor noise while maintaining the sharp step-discontinuities characteristic of pasted donor edges.

---

### 2.4 High-Pass Spatial Edge Derivatives: Sobel Gradients & Laplacian

#### First-Order Directional Sobel Gradients
Directional derivative approximations using $3 \times 3$ convolution masks:
$$\mathbf{G}_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I, \quad \mathbf{G}_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I$$
Spatial Gradient Magnitude:
$$M(x, y) = \sqrt{\mathbf{G}_x^2(x, y) + \mathbf{G}_y^2(x, y)}$$
Gradient Direction:
$$\theta(x, y) = \arctan\left( \frac{\mathbf{G}_y(x, y)}{\mathbf{G}_x(x, y)} \right)$$

#### Second-Order Isotropic Derivative: The Laplacian
$$\nabla^2 I(x, y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
Discrete isotropic kernel:
$$\mathbf{L} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix} \quad \text{or} \quad \mathbf{L}_8 = \begin{bmatrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$
*Forensic Utility:* 1st-order gradients highlight ramp edges; 2nd-order Laplacian zeroes out flat areas and highlights isolated points and double-edge transitions common along pasted seams.

---

### 2.5 Local Statistical Variance Gating

In feature-based forensics (SIFT/ORB), textureless regions (skies, walls) yield degenerate descriptors. We compute local statistical variance over window $W$ of size $(2k+1) \times (2k+1)$:
$$\sigma^2(x, y) = \frac{1}{|W|} \sum_{(u, v) \in W} I(x+u, y+v)^2 - \left( \frac{1}{|W|} \sum_{(u, v) \in W} I(x+u, y+v) \right)^2$$
Any keypoint located where $\sigma^2(x, y) < \tau_{\text{flat}}$ is pruned prior to descriptor matching.

---

## 3. Module 2: Frequency Domain Analysis & Homomorphic Filtering (Chapter 4)

Spatial convolution corresponds to simple multiplication in the frequency domain:
$$f(x, y) * h(x, y) \Longleftrightarrow F(u, v) \cdot H(u, v)$$

### 3.1 2D Discrete Fourier Transform (DFT) & Centering Property

#### Continuous 2D Fourier Integral
$$F(u, v) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) e^{-j 2\pi (ux + vy)} \,dx \,dy$$

#### 2D Discrete Fourier Transform (DFT)
For an $M \times N$ discrete digital image:
$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \exp\left[ -j 2\pi \left( \frac{ux}{M} + \frac{vy}{N} \right) \right]$$
Inverse 2D DFT:
$$f(x, y) = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u, v) \exp\left[ +j 2\pi \left( \frac{ux}{M} + \frac{vy}{N} \right) \right]$$

#### Centering the Spectrum
By multiplying $f(x, y)$ by $(-1)^{x+y}$ in spatial domain, the zero-frequency DC component $F(0, 0)$ is shifted to the geometric center of the frequency rectangle $(M/2, N/2)$:
$$\mathcal{F}\{ f(x, y) \cdot (-1)^{x+y} \} = F\left( u - \frac{M}{2}, v - \frac{N}{2} \right)$$

#### Logarithmic Fourier Magnitude Spectrum
$$|F(u, v)| = \sqrt{ \text{Re}\{F(u, v)\}^2 + \text{Im}\{F(u, v)\}^2 }$$
$$S(u, v) = c \cdot \ln(1 + |F(u, v)|)$$

---

### 3.2 Ideal, Gaussian, and Butterworth Transfer Functions

Let $D(u, v)$ represent the Euclidean distance from frequency coordinates $(u, v)$ to the center $(M/2, N/2)$:
$$D(u, v) = \sqrt{\left( u - \frac{M}{2} \right)^2 + \left( v - \frac{N}{2} \right)^2}$$

#### Transfer Function Formulations:
| Filter Type | Low-Pass Filter $H_{\text{LP}}(u, v)$ | High-Pass Filter $H_{\text{HP}}(u, v)$ | Ringing Artifacts |
| :--- | :--- | :--- | :--- |
| **Ideal** | $1 \text{ if } D(u, v) \le D_0 \text{ else } 0$ | $1 - H_{\text{ILPF}}(u, v)$ | Severe (Gibbs phenomenon) |
| **Gaussian** | $\exp\left( -\frac{D^2(u, v)}{2 D_0^2} \right)$ | $1 - \exp\left( -\frac{D^2(u, v)}{2 D_0^2} \right)$ | None (Smooth transition) |
| **Butterworth** (Order $n$) | $\frac{1}{1 + \left[ D(u, v) / D_0 \right]^{2n}}$ | $\frac{1}{1 + \left[ D_0 / D(u, v) \right]^{2n}}$ | Tunable via order $n$ |

---

### 3.3 The Illumination-Reflectance Model & Homomorphic Filtering

An image is physically generated by the product of illumination $i(x, y)$ and reflectance $r(x, y)$:
$$f(x, y) = i(x, y) \cdot r(x, y)$$
- **Illumination $i(x, y)$:** Light incident on the scene; characterized by slow spatial variations $\implies$ **Low spatial frequencies**.
- **Reflectance $r(x, y)$:** Object physical surface reflectivity and sharp structural boundaries $\implies$ **High spatial frequencies**.

#### The Decoupling Problem:
Because $i$ and $r$ are multiplicative, standard linear Fourier filtering cannot separate them:
$$\mathcal{F}\{f\} = \mathcal{F}\{i \cdot r\} \neq \mathcal{F}\{i\} \cdot \mathcal{F}\{r\}$$

#### Homomorphic Filtering Pipeline:
```
 f(x, y) ---> [ ln(f + eps) ] ---> [ 2D FFT ] ---> [ H(u, v) ] ---> [ 2D IFFT ] ---> [ exp(s) ] ---> g(x, y)
                z(x, y)             Z(u, v)         S(u, v)          s(x, y)
```

1. **Logarithmic Transformation (Multiplicative $\to$ Additive):**
   $$z(x, y) = \ln(f(x, y) + \epsilon) = \ln(i(x, y)) + \ln(r(x, y))$$
2. **2D Fast Fourier Transform:**
   $$Z(u, v) = \mathcal{F}\{z(x, y)\} = I(u, v) + R(u, v)$$
3. **Modified High-Emphasis Butterworth Filter:**
   $$H(u, v) = (\gamma_H - \gamma_L) \left[ 1 - \exp\left( -c \frac{D^2(u, v)}{D_0^2} \right) \right] + \gamma_L$$
   Where $\gamma_L < 1.0$ (attenuates illumination low frequencies) and $\gamma_H > 1.0$ (amplifies reflectance high frequencies).
4. **Inverse 2D FFT:**
   $$s(x, y) = \mathcal{F}^{-1}\{ H(u, v) \cdot Z(u, v) \}$$
5. **Exponential Inversion:**
   $$g(x, y) = \exp(s(x, y)) - \epsilon$$

#### Forensic Application to Splice Detection:
Spliced foreign donor patches typically originate from a different environment with distinct incident light direction and intensity. Decoupled illumination residual:
$$\Delta_{\text{illum}}(x, y) = |f(x, y) - g(x, y)|$$
exhibits sharp step changes along spliced boundaries due to mismatched lighting gradients.

---

## 4. Module 3: Compression Redundancies & Information Entropy (Chapter 8)

### 4.1 JPEG Compression Pipeline & Quantization Matrices

The lossy JPEG pipeline consists of:
1. Division of image into non-overlapping $8 \times 8$ pixel blocks.
2. 2D Discrete Cosine Transform (DCT) on each block:
   $$F_{\text{DCT}}(u, v) = \frac{1}{4} C(u) C(v) \sum_{x=0}^7 \sum_{y=0}^7 f(x, y) \cos\left[ \frac{(2x+1)u\pi}{16} \right] \cos\left[ \frac{(2y+1)v\pi}{16} \right]$$
3. Lossy Quantization by standard Luminance Quantization Matrix $Q$:
   $$F_Q(u, v) = \text{round}\left( \frac{F_{\text{DCT}}(u, v)}{Q(u, v)} \right)$$
4. Lossless Huffman Run-Length entropy coding.

---

### 4.2 Error Level Analysis (ELA) Physics & Equilibrium

#### Theoretical Principle:
When an authentic JPEG is saved repeatedly at a reference quality factor $Q_{\text{ref}}$, its DCT coefficients reach an **information equilibrium** where subsequent quantization produces negligible residual error:
$$|f_{Q_{\text{ref}}}(x, y) - f_{Q_{\text{ref}}}^{(2)}(x, y)| \approx 0$$
However, if a spliced element (either uncompressed PNG or from a donor with quality $Q_{\text{donor}} \neq Q_{\text{host}}$) is inserted:
$$E_{\text{raw}}(x, y) = |f(x, y) - f_{Q_{\text{ref}}}(x, y)| \gg 0$$
The spliced foreign patch exhibits elevated error variance because its quantization matrix history differs from the host canvas.

---

### 4.3 8×8 Block Artifact Grid (BAG) Step-Discontinuity Detection

Due to independent block-by-block quantization, authentic JPEG images develop subtle boundary steps exactly at coordinate multiples of 8:
$$\{(x, y) \mid x \pmod 8 = 0 \text{ or } y \pmod 8 = 0\}$$
When a patch is pasted into the host canvas without exact 8-pixel grid phase alignment, it cuts across the native 8×8 blocks.

#### Boundary Discontinuity Operator:
We evaluate 2nd-order finite differences across adjacent pixels:
$$\mathbf{D}_h(x, y) = |2 I(x, y) - I(x+1, y) - I(x-1, y)|$$
$$\mathbf{D}_v(x, y) = |2 I(x, y) - I(x, y+1) - I(x, y-1)|$$
$$\text{BAG}(x, y) = \frac{\mathbf{D}_h(x, y) + \mathbf{D}_v(x, y)}{2} * W_{8 \times 8}$$
A localized disruption in the periodic 8-pixel grid periodicity confirms an unaligned splice.

---

### 4.4 Local Shannon Information Entropy Mapping

From Shannon’s Information Theory, average information content (entropy) within spatial window $W$ is:
$$H(x, y) = -\sum_{k=0}^{L-1} p(r_k) \log_2(p(r_k))$$
- Natural camera sensor noise: High entropy ($H \to 7.5 - 8.0$ bits).
- Computer-generated cloning, heavy quantization, or flat pasted patches: Low entropy ($H \ll 5.0$ bits).

---

## 5. Module 4: Forensic Detection Orchestration (Chapters 10 & 11)

### 5.1 Copy-Move Detection: Keypoints, g2NN Matching & RANSAC

1. **Feature Detection:** Extract keypoint coordinates $P = \{p_i\}$ and descriptors $D = \{d_i\}$ using SIFT (Scale-Invariant Feature Transform) / ORB:
   $$\text{SIFT: } 128\text{-dimensional float descriptor}, \quad \text{ORB: } 256\text{-bit binary descriptor}$$
2. **Variance Gating:** Discard keypoints with $\sigma^2(p_i) < \tau_{\text{flat}}$.
3. **Generalized 2NN (g2NN) Distance Ratio:**
   Rank descriptor Euclidean distances $d_{(1)} \le d_{(2)} \le \dots \le d_{(k)}$:
   $$\frac{\|d_i - d_{(1)}\|_2}{\|d_i - d_{(2)}\|_2} < \rho \quad (\rho = 0.65)$$
   Subject to minimum spatial displacement:
   $$\|p_i - p_{(1)}\|_2 > \delta_{\text{min}} \quad (\delta_{\text{min}} = 20\text{ px})$$
4. **RANSAC Rigid / Affine Model Verification:**
   Fit affine transformation matrix $\mathbf{H} \in \mathbb{R}^{2 \times 3}$:
   $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} \\ h_{21} & h_{22} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} h_{13} \\ h_{23} \end{bmatrix}$$
   Points satisfying $\|p'_i - \mathbf{H} p_i\|_2 < \epsilon_{\text{reproj}}$ are classified as verified inliers. If $N_{\text{inliers}} \ge N_{\text{thresh}}$, copy-move cloning is confirmed.

---

### 5.2 Splice Detection: Multi-Cue Convex Combination

Composite anomaly heatmap $A(x, y)$ fuses compression, frequency, and block cues:
$$A(x, y) = w_{\text{ela}} \cdot E_{\text{norm}}(x, y) + w_{\text{homo}} \cdot \Delta_{\text{illum}}(x, y) + w_{\text{bag}} \cdot \text{BAG}(x, y)$$
$$\text{subject to } w_{\text{ela}} + w_{\text{homo}} + w_{\text{bag}} = 1.0, \quad w_i \ge 0$$

---

## 6. Module 5: Mathematical Morphology & Mask Refinement (Chapters 9 & 10)

Raw anomaly heatmaps $A(x, y) \in [0.0, 1.0]$ are noisy and continuous. Mathematical morphology converts them into crisp, topologically solid binary tamper masks $g(x, y) \in \{0, 255\}$.

### 6.1 Set-Theoretic Operators: Erosion, Dilation, Opening, Closing

Let $A$ represent foreground pixels and $B$ represent a structuring element (SE):

#### Erosion (Contraction):
$$(A \ominus B) = \{z \mid (B)_z \subseteq A\}$$
Removes isolated noise pixels smaller than $B$.

#### Dilation (Expansion):
$$(A \oplus B) = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}$$
Bridges broken segments and fills interior voids.

#### Morphological Opening:
$$A \circ B = (A \ominus B) \oplus B$$
Suppresses small foreground protrusions without altering large blob geometry.

#### Morphological Closing:
$$A \bullet B = (A \oplus B) \ominus B$$
Fuses narrow breaks and bridges contiguous contours.

---

### 6.2 Geodesic Hole-Filling & Morphological Reconstruction

Iterative conditional dilation of marker image $F$ inside mask $G$:
$$D_G^{(1)}(F) = (F \oplus B) \cap G$$
Repeated until convergence:
$$D_G^{(k)}(F) = D_G^{(1)}[D_G^{(k-1)}(F)] = D_G^{(k-1)}(F)$$
*Forensic Utility:* Seals non-tampered interior voids inside a detected cloned/spliced object, ensuring a solid segmentation mask.

---

### 6.3 Otsu’s Optimal Between-Class Variance Thresholding

To convert continuous heatmap $A(x, y)$ to binary mask, Otsu’s algorithm automatically calculates threshold $t^*$ that maximizes between-class variance $\sigma_B^2(t)$:
1. Pixel probabilities: $p_i = n_i / N$
2. Class probabilities for background $C_0$ and foreground $C_1$:
   $$\omega_0(t) = \sum_{i=0}^t p_i, \quad \omega_1(t) = 1 - \omega_0(t)$$
3. Class means:
   $$\mu_0(t) = \sum_{i=0}^t \frac{i \cdot p_i}{\omega_0(t)}, \quad \mu_1(t) = \sum_{i=t+1}^{L-1} \frac{i \cdot p_i}{\omega_1(t)}$$
4. Between-class variance optimization:
   $$t^* = \arg\max_{0 \le t < L-1} \left\{ \omega_0(t) \omega_1(t) \left[ \mu_0(t) - \mu_1(t) \right]^2 \right\}$$
5. Binarization:
   $$g_{\text{raw}}(x, y) = \begin{cases} 255, & \text{if } A(x, y) \ge t^* \\ 0, & \text{otherwise} \end{cases}$$

---

### 6.4 Connected Component Analysis (CCA) & Skeleton Boundary Extraction

#### Connected Component Analysis (CCA)
1. Segment binary mask into 8-connected disjoint components:
   $$\mathcal{C} = \{C_1, C_2, \dots, C_K\}$$
2. Compute area (pixel count) $\text{Area}(C_k) = \sum_{(x, y) \in C_k} 1$.
3. Gating rule:
   $$g(x, y) = \begin{cases} 255, & \text{if } (x, y) \in C_k \text{ and } \text{Area}(C_k) \ge \theta_{\text{min}} \\ 0, & \text{otherwise} \end{cases}$$
   Eliminates isolated false-positive clusters ($\theta_{\text{min}} = 150\text{ px}$).

#### Morphological Boundary Extraction
$$\beta(g) = g - (g \ominus B_{3 \times 3})$$
Produces a crisp 1-pixel boundary tracing the exact perimeter of the forged region for overlay visualization.

---

## 7. End-to-End Mathematical Transformation Matrix

$$\begin{aligned}
\mathbf{f}(x, y) \in \mathbb{R}^{H \times W \times 3} &\xrightarrow{\text{Luminance Conversion}} I(x, y) \in [0, 255] \\
I(x, y) &\xrightarrow{\text{Log Transform}} z(x, y) = \ln(I(x, y) + \epsilon) \\
z(x, y) &\xrightarrow{\text{2D FFT}} Z(u, v) = \mathcal{F}\{z(x, y)\} \\
Z(u, v) &\xrightarrow{\text{Butterworth High-Emphasis}} S(u, v) = H(u, v) \cdot Z(u, v) \\
S(u, v) &\xrightarrow{\text{2D IFFT \& Exponential}} \hat{r}(x, y) = \exp(\mathcal{F}^{-1}\{S\}) - \epsilon \\
\mathbf{f}(x, y) &\xrightarrow{\text{JPEG Re-compression } Q=90} \mathbf{f}_Q(x, y) \\
E_{\text{raw}}(x, y) &= \|\mathbf{f}(x, y) - \mathbf{f}_Q(x, y)\|_2 \\
A(x, y) &= w_1 E_{\text{norm}} + w_2 |I - \hat{r}| + w_3 \text{BAG}(I) \\
g_{\text{raw}}(x, y) &= \mathbb{I}[A(x, y) \ge t_{\text{Otsu}}] \\
g(x, y) &= \text{CCA}_{\theta}\left( \text{FillHoles}\left( (g_{\text{raw}} \circ B_{\text{open}}) \bullet B_{\text{close}} \right) \right) \in \{0, 255\}
\end{aligned}$$

---

## 8. Conclusion & Viva / Examination Quick Reference

| Viva Question | Core DIP Answer |
| :--- | :--- |
| **Why not use Deep Learning?** | Deep learning is a black-box that hallucinates on unseen compression levels. Classical DIP is deterministic, fully explainable, and mathematically provable. |
| **Why do we need Homomorphic Filtering?** | Illumination and reflectance are multiplicative ($f = i \cdot r$). Linear filters cannot separate them. Logarithm converts multiplication into addition: $\ln(f) = \ln(i) + \ln(r)$. |
| **How does ELA detect spliced images?** | Authentic JPEG images reach a quantization equilibrium at $Q_{\text{ref}}$. Spliced donor patches from different quality levels display elevated error residual variance. |
| **Why use g2NN instead of standard 2NN?** | Standard 2NN ratio fails when an object is cloned multiple times because nearest neighbors have identical descriptors. Generalized 2NN compares matches against subsequent distinct neighbors with spatial displacement constraints. |
| **Why is Morphological Opening performed before Closing?** | Opening ($(A \ominus B) \oplus B$) eliminates isolated salt-and-pepper noise spikes first. Closing ($(A \oplus B) \ominus B$) then bridges remaining genuine component gaps without welding noise into the mask. |
