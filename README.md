<div align="center">
  <img src="docs/logo.png" alt="S.P.E.C.T.R.A. Logo" width="180"/>
  <h1>S.P.E.C.T.R.A.</h1>
  <p><strong>Spatial Processing and Entropy Checks for Tamper Recognition Analysis</strong></p>
  <p><em>A 100% Classical Digital Image Processing (DIP) Framework for Copy-Move &amp; Splice Forensics</em></p>

  <p>
    <a href="https://github.com/Kirit0-00/S.P.E.C.T.R.A"><img src="https://img.shields.io/badge/GitHub-Kirit0--00%2FS.P.E.C.T.R.A-181717.svg?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repository"/></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/></a>
    <a href="https://docs.opencv.org/"><img src="https://img.shields.io/badge/OpenCV-5.0%2B-5C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/></a>
    <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-Vectorized-013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/></a>
    <a href="https://scipy.org/"><img src="https://img.shields.io/badge/SciPy-FFT%20%26%20Filters-8CAAE6.svg?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy"/></a>
    <a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-Plotting-11557c.svg?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib"/></a>
    <a href="https://scikit-image.org/"><img src="https://img.shields.io/badge/scikit--image-Morphology-orange.svg?style=for-the-badge" alt="scikit-image"/></a>
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-UI%20Dashboard-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/></a>
    <a href="https://docs.pytest.org/"><img src="https://img.shields.io/badge/PyTest-28%20Passed-0A9EDC.svg?style=for-the-badge&logo=pytest&logoColor=white" alt="PyTest"/></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-brightgreen.svg?style=for-the-badge" alt="License: MIT"/></a>
  </p>
</div>

---

## 📖 Overview

**S.P.E.C.T.R.A.** (**S**patial **P**rocessing and **E**ntropy **C**hecks for **T**amper **R**ecognition **A**nalysis) is a modular, mathematically rigorous image forensics suite designed to detect and localize digital image tampering — specifically **Copy-Move (Cloning)** and **Image Splicing (Compositing)**.

Unlike black-box deep learning models that suffer from dataset hallucination and lack explainability, **S.P.E.C.T.R.A.** operates entirely on classical principles from Digital Image Processing (Gonzalez & Woods). Every detection stage transforms raw spatial image data through frequency decomposition, compression artifact analysis, and mathematical morphology to produce explainable binary tamper heatmaps.

> 📘 **Mathematical & Theoretical Course Manual:**
> For in-depth mathematical derivations, formula proofs, and exam/viva explanations mapped directly to **Gonzalez & Woods**, see [**`DIP_THEORY_AND_ARCHITECTURE.md`**](DIP_THEORY_AND_ARCHITECTURE.md).

---

## 🕵️ What is Copy-Move Forgery?

**Copy-Move** (also called *cloning*) is when someone **copies a region from within the same photo and pastes it somewhere else** in the same image to hide or duplicate something.

### Real-world examples:
- Duplicating trees in a landscape photo to hide a person behind them
- Cloning a crowd section to make an event look more crowded
- Copying part of a background to cover an unwanted object

### How it looks:
```
Original:        [Sky][Tree][Person][Grass]
After Copy-Move: [Sky][Tree][Tree  ][Grass]   ← Person is gone, Tree duplicated
```

### Why it's hard to see:
The copied patch came from **the same image**, so the lighting, noise, and camera fingerprint all match perfectly. The human eye often can't tell.

### How S.P.E.C.T.R.A. catches it:
| Technique | What it detects |
|-----------|----------------|
| **SIFT + RANSAC** | Two spatially-separated regions have nearly identical 128-dim keypoint descriptors → geometric affine match confirms copy |
| **DCT Block Matching** | The copied 16×16 pixel blocks produce identical DCT coefficient vectors → flagged as duplicates |
| **ELA** | The pasted patch has a slightly different JPEG quantisation history than its new surroundings → shows as anomalous residual |

---

## ✂️ What is Image Splicing?

**Image Splicing** is when a region from a **completely different photo** is cut and pasted into the host image to fabricate a scene that never existed.

### Real-world examples:
- Placing a politician into a crowd at an event they never attended
- Adding a UFO to a sky photo taken from a different image
- Swapping a person's face with someone else's from another photo

### How it looks:
```
Donor image:  [Different camera, different light, different JPEG compression]
Host image:   [Original scene]
After Splice: [Original scene] + [Foreign patch from donor]  ← looks seamless but isn't
```

### Why it's hard to see:
Modern tools (Photoshop, GIMP) can blend seams with feathering, color-matching, and blending modes. But the **physics of JPEG compression and sensor noise** leave invisible fingerprints that reveal the foreign origin.

### How S.P.E.C.T.R.A. catches it:
| Technique | What it detects |
|-----------|----------------|
| **ELA (Error Level Analysis)** | Donor compressed at different quality → different JPEG equilibrium → bright anomaly on re-encode |
| **Homomorphic Filtering** | Donor has different lighting model (illuminance × reflectance) → residual mismatch in log-frequency domain |
| **Block Artifact Grid (BAG)** | Donor's 8×8 JPEG grid is misaligned with host's grid → discontinuity at paste boundaries |
| **Shannon Entropy** | Foreign regions have different local randomness pattern than the host's natural noise |

---

## 🔬 Tech Stack & Official Documentation

This project enforces a zero-deep-learning policy and is built exclusively upon classical scientific Python libraries:

- [**OpenCV (cv2)**](https://docs.opencv.org/): Spatial transformations, bilateral filtering, CLAHE, and keypoint matching.
- [**NumPy**](https://numpy.org/doc/stable/): Vectorized matrix operations, geometric array manipulations, and statistical variance computation.
- [**SciPy**](https://docs.scipy.org/doc/scipy/): Fast Fourier Transform (FFT) optimization via `scipy.fft` and 2D spatial filtering.
- [**Matplotlib**](https://matplotlib.org/stable/): Scientific spectrum visualizations and color-coded overlay plots.
- [**scikit-image**](https://scikit-image.org/docs/stable/): Local Shannon rank entropy and morphological geodesic reconstruction.
- [**Streamlit**](https://docs.streamlit.io/): Interactive web dashboard providing stage-by-stage visual inspection tabs.

---

## 📚 Features (Mapped to DIP Syllabus)

The architecture is partitioned into five distinct modules directly corresponding to classical DIP curriculum units:

### 1. Spatial Preprocessing (`src/spatial/`)
- **Intensity Transformations:** Logarithmic compression ($s = c \ln(1 + r)$) and power-law gamma correction ($s = c r^\gamma$).
- **Contrast Enhancement:** Global Histogram Equalization and Contrast-Limited Adaptive Histogram Equalization (CLAHE) to expose subtle tampering borders.
- **Spatial Filtering:** 2D Gaussian smoothing, edge-preserving Bilateral filtering, 1st-order Sobel directional derivatives ($G_x, G_y$), isotropic Laplacian ($\nabla^2 f$), High-Boost unsharp masking, and local statistical variance ($\sigma^2(x, y)$) for flat-region gating.

### 2. Frequency Domain Analysis (`src/frequency/`)
- **2D Fast Fourier Transform:** Centered 2D FFT/IFFT with optimal composite zero-padding.
- **Frequency Filtering:** Ideal, Gaussian, and Butterworth Low-Pass (LPF) and High-Pass (HPF) transfer functions.
- **Homomorphic Filtering:** Decoupling illumination $i(x, y)$ and reflectance $r(x, y)$ via logarithmic Fourier transformation ($z = \ln(f + \epsilon) \to Z \to H \cdot Z \to s \to \exp(s)$) to isolate lighting inconsistencies in foreign donor splices.

### 3. Compression & Information Entropy (`src/compression/`)
- **Error Level Analysis (ELA):** Analyzes quantization matrices by re-encoding images at a fixed reference quality ($Q_{\text{ref}} = 90$), highlighting non-equilibrium residual variance in foreign patches.
- **Block Artifact Grid (BAG):** Horizontal and vertical second-difference filtering ($[1, -2, 1]$) to detect disruptions in JPEG's 8×8 DCT grid.
- **Local Shannon Entropy:** Computes local information entropy $H(x, y) = -\sum p(i) \log_2 p(i)$ to separate natural sensor noise from manipulated regions.

### 4. Forensic Detection Orchestration (`src/detection/`)
- **Copy-Move Analysis:** SIFT/ORB feature detection with variance filtering, Generalized 2-Nearest-Neighbor (g2NN) distance ratio matching, and RANSAC affine transformation verification.
- **Splice Detection:** Multi-cue anomaly fusion combining ELA residuals, homomorphic illumination gradients, and block boundary step discontinuities.

### 5. Morphological Post-Processing (`src/morphology/`)
- **Morphological Cleaning:** Structuring-element erosion, dilation, opening (noise removal), and closing (gap bridging).
- **Segmentation & Refinement:** Otsu's optimal thresholding, geodesic hole filling, Connected Component Analysis (CCA) size gating, and skeleton boundary extraction.

---

## 🧪 Verified on Real Internet Image

The tool has been tested on a real Canon EOS 400D DSLR photograph (Wikimedia Commons), not just synthetic test images.

**Image:** Ant macro photograph (600×400 px, natural JPEG, sensor noise present)

| Test | Result |
|------|--------|
| Copy-Move (SIFT+RANSAC) | ✅ detected, **38 inliers** in 0.196 s |
| Copy-Move (DCT Blocks) | ✅ detected, heatmap_max = 1.0000 |
| Splice (Fused heatmap) | ✅ **20.98% pixels flagged** (50,363 px) |
| ELA splice vs clean | 0.008965 vs 0.002533 (3.5× difference) |
| BAG splice vs clean | 0.1418 vs 0.0093 (15× difference) |

See [`LIVE_EXPERIMENTS.md`](LIVE_EXPERIMENTS.md) for the full step-by-step guide with download instructions and expected output.

---

## 🚀 Installation

Ensure Python 3.10 or higher is installed on your system.

```bash
# 1. Clone the repository
git clone https://github.com/Kirit0-00/S.P.E.C.T.R.A.git
cd S.P.E.C.T.R.A

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install opencv-python-headless numpy scipy scikit-image matplotlib streamlit pytest
```

---

## 💻 Usage

### Running Unit Tests & Forensic Pipeline Verification
Execute the automated test suite across all DIP modules:

```bash
pytest tests/
# Expected: 28 passed
```

### Launching the Interactive Streamlit Dashboard
Launch the web interface to analyze your own images or run automated synthetic test cases:

```bash
streamlit run app.py
# Opens at: http://localhost:8501
```

### CLI — Analyze Any Image

```bash
# Analyze a real image file
python cli.py --image /path/to/image.jpg --mode full

# Run built-in synthetic scenarios
python cli.py --synthetic copy_move --json
python cli.py --synthetic splice --json
python cli.py --synthetic authentic --json
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
