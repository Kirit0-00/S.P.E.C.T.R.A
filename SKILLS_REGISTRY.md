# DIP Forensics Custom Skills Registry

This registry formalizes specialized analytical workflows for testing, verifying, and tuning classical image forensics pipelines.

---

## Skill 1: `DIP_Syllabus_Compliance_Check`
- **Purpose:** Audit any proposed or generated Python file to guarantee zero deep learning imports and verify that all transformations match formal DIP textbook definitions (Gonzalez & Woods).
- **Trigger:** Prior to finalizing any `.py` module or PR.
- **Inputs:** Target file path (e.g., `src/spatial/filters.py`).
- **Procedure:**
  1. AST inspection for forbidden imports (`torch`, `tensorflow`, `keras`, `onnx`, `sklearn.neural_network`).
  2. Line count verification (`wc -l <= 200`).
  3. Mathematical annotation verification (ensure documentation links algorithms to spatial/frequency/compression/morphological theory).
- **Output:** Compliance Pass/Fail Report with line-by-line violation trace.

---

## Skill 2: `Forensic_Mask_Evaluator`
- **Purpose:** Objectively evaluate predicted binary tamper masks against synthetic or ground-truth masks using standard DIP segmentation metrics.
- **Trigger:** Post-execution of tamper detection pipelines on benchmark datasets or synthetic testbeds.
- **Inputs:**
  - `pred_mask`: Binary mask `(H, W)` $\in \{0, 255\}$.
  - `gt_mask`: Ground-truth binary mask `(H, W)` $\in \{0, 255\}$.
- **Metrics Computed:**
  - $Precision = \frac{TP}{TP + FP}$
  - $Recall = \frac{TP}{TP + FN}$
  - $F_1\text{-Score} = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
  - $IoU = \frac{|pred \cap gt|}{|pred \cup gt|}$
  - Morphological Boundary Error (Hausdorff distance on contour skeletons).
- **Output:** Formatted metric scorecard with visual overlay plot (TP: Green, FP: Red, FN: Blue).

---

## Skill 3: `Dataset_Threshold_Tuner`
- **Purpose:** Empirically calibrate dynamic threshold parameters (e.g., CLAHE clip limit, ELA quality factor $\Delta Q$, Otsu offset factor $\alpha$, homomorphic filter cutoff $D_0$) across synthetic variations.
- **Trigger:** When detection precision drops below $0.80$ on synthetic benchmark runs.
- **Inputs:** Calibration batch of forged vs authentic images with varied compression artifacts.
- **Procedure:**
  1. Sweep parameter spaces:
     - Homomorphic $\gamma_H \in [1.2, 2.5]$, $\gamma_L \in [0.2, 0.8]$, $D_0 \in [10, 80]$.
     - ELA compression resave quality $Q \in [75, 95]$.
     - Morphological structural element size $k \in [3, 11]$.
  2. Compute ROC / PR curves.
  3. Select optimal operating point (maximizing $F_1$ score).
- **Output:** Calibrated configuration dictionary (`config.json` or YAML).
