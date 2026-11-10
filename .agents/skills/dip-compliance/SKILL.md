---
name: dip-compliance
description: Audits code for classical DIP compliance (Gonzalez & Woods), zero deep-learning dependencies, and file size limits (<= 200 lines).
---

# DIP Syllabus Compliance Skill

## Objective
Enforce strict classical Digital Image Processing (DIP) constraints on all image forensics code.

## Verification Checklist
1. **Forbidden Dependencies:** Zero imports of `torch`, `tensorflow`, `keras`, `onnx`, or black-box neural networks.
2. **Permitted Stack:** `numpy`, `scipy`, `cv2` (OpenCV), `skimage` (scikit-image), `matplotlib`, `streamlit`.
3. **Module Size:** Every `.py` file must contain $\le 200$ lines.
4. **Syllabus Integrity:** Every function must map to classical spatial, frequency, compression, or morphological processing principles.
