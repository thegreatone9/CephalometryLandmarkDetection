# Complete Literature Map: AI Cephalometric Landmark Detection (2023–2026)

Every paper your reviewers might compare you to, organized by venue tier.

---

## Tier 1: Top Conferences (MICCAI, ECCV)

### 1. D-CeLR — Dual-Encoder Cephalometric Landmark Regression
- **Venue:** ECCV 2024
- **Method:** Dual-encoder (coarse → fine), end-to-end differentiable regression
- **Dataset:** ISBI 2015 benchmark (400 images, 19 landmarks)
- **Results:** SOTA on ISBI 2015 benchmark; outperforms cascading models with fewer params
- **Code:** [github.com/huang229/D-CeLR](https://github.com/huang229/D-CeLR)
- **Relevance to you:** Different paradigm (coordinate regression, not heatmap). Different dataset. You cite as "alternative approach" in Related Work.

### 2. CeLDA — Cephalometric Landmark Detection across Ages
- **Venue:** MICCAI 2024
- **Method:** Prototypical network with prototype alignment for adolescent + adult subjects
- **Dataset:** Custom "CephAdoAdu" (adolescent + adult images)
- **Results:** SOTA on their custom benchmark across age groups
- **Code:** [github.com/ShanghaiTech-IMPACT/CeLDA](https://github.com/ShanghaiTech-IMPACT/CeLDA)
- **Relevance to you:** Different problem (age generalization). Cite in Related Work as "domain adaptation" approach.

### 3. CL-Detection 2024 Challenge Winner — "Optimising for the Unknown"
- **Venue:** MICCAI 2024 Challenge
- **Method:** ConvNeXt V2 + RCNN, domain alignment, X-ray artifact augmentation
- **Dataset:** 700 images from 4 medical centers, 53 landmarks
- **Results:** **MRE = 1.186 mm, SDR@2mm = 82.04%**
- **Winner:** Julian Wyatt, University of Oxford
- **Relevance to you:** This is the current MICCAI SOTA. Your SDR on 6 landmarks (94.9%) exceeds this, but they detect 53 landmarks across 4 domains. Your 29-landmark results will be directly comparable.

### 4. CL-Detection 2023 Challenge
- **Venue:** MICCAI 2023 Challenge
- **Dataset:** 600 images from 3 medical centers, 38 landmarks
- **Results:** Top entries achieved MRE ~1.2–1.5 mm range
- **Relevance to you:** Historical context. Cite as prior iteration of the benchmark.

---

## Tier 2: Journals (Nature, Springer, MDPI, BMC)

### 5. YOLOv12-Based Cephalometric Landmark Detection ⭐
- **Venue:** Nature Scientific Reports (March 2026)
- **Method:** YOLOv12-l bounding box detection, 19 landmarks
- **Dataset:** 499 images from Roboflow (2 merged sources)
- **Results:** **SDR@2mm = 80.57%**, no MRE reported
- **Relevance to you:** **Primary comparison target.** Same venue you're targeting. Your heatmap method is architecturally superior to their bounding box approach.

### 6. CBCT U-Net + Efficient Global Attention
- **Venue:** Nature Scientific Reports (February 2026)
- **Method:** Optimized U-Net + attention module, 3D CBCT
- **Dataset:** 75 CBCT scans
- **Results:** 59.15% of measurements clinically interchangeable (±2.0 mm/°)
- **Relevance to you:** Different modality (3D CBCT vs your 2D). Cite as "3D extension" in Discussion but not a direct comparison.

### 7. AI vs Manual Cephalometric Evaluation
- **Venue:** Head & Face Medicine (March 2026)
- **Method:** Evaluation study of commercial AI tools vs manual tracing
- **Results:** Confirmed AI reaches clinical-level reproducibility
- **Relevance to you:** Cite for clinical context — supports your motivation that AI cephalometry is clinically viable.

### 8. CephaNN — Automated Cephalometric Points Marking System
- **Venue:** MDPI (May 2026)
- **Method:** Ensemble of two U-Net subnetworks + region-enhancing (RE) loss
- **Results:** Improved SDR over single-model baselines
- **Relevance to you:** Directly comparable — same architecture family (U-Net + heatmap). Cite in Related Work.

### 9. U-Net Landmark Detection in Cephalometric Images
- **Venue:** AIP Conference Proceedings (March 2026)
- **Method:** U-Net + multichannel heatmap regression, 19 landmarks
- **Relevance to you:** Almost identical method to yours but on a different dataset with fewer landmarks. Validates that U-Net + heatmap is still publishable.

### 10. Aariz Dataset Paper ⭐
- **Venue:** Nature Scientific Data (July 2025)
- **Authors:** Khalid, M. A., et al.
- **Content:** 1000 lateral cephalograms, 29 landmarks, 7 X-ray machines, CVM stage labels
- **Relevance to you:** **You must cite this.** It's the dataset you use. Their paper includes baseline DL results you can compare against.

### 11. DeepFuse — Multimodal Framework
- **Venue:** Nature Scientific Reports (July 2025) — **RETRACTED March 2026**
- **Method:** Fusion of cephalograms + CBCT + dental models
- **Results:** MRE = 1.21 mm (claimed)
- **Relevance to you:** Cite as cautionary example in Discussion about research integrity. Do NOT compare your numbers against retracted results.

---

## Tier 3: Preprints & Withdrawn Papers

### 12. CephRes-MHNet — Multi-Head Residual Network ⭐
- **Venue:** arXiv (Nov 2025) — **WITHDRAWN** (authorization issues)
- **Method:** Multi-head residual CNN, 4.7M params
- **Dataset:** **Aariz (1000 images, 29 landmarks)**
- **Results:** **MRE = 1.23 mm, SDR@2mm = 85.5%**
- **Relevance to you:** **Most directly comparable.** Same dataset, same landmarks. Their numbers are your benchmark to beat. But since it's withdrawn, you can cite cautiously ("concurrent unpublished work").

### 13. YOLOv11 Clinical Validation
- **Venue:** Preprints.org (May 2026, under review)
- **Method:** YOLOv5/v11 variants, 4 landmarks only (S, N, A, B)
- **Results:** MRE = 3.10 mm, SDR@4mm = 87.2%
- **Relevance to you:** Weak results (3.10mm MRE is poor). Cite as example of YOLO limitations for precise landmarking.

### 14. Diffusion-Based Data Generation for Cephalometric Detection
- **Venue:** arXiv (May 2025)
- **Method:** Diffusion model generates synthetic X-rays with anatomical priors
- **Results:** +6.5% SDR improvement, SDR@2mm = 82.2%
- **Relevance to you:** Different contribution type (data augmentation, not architecture). Cite in Related Work as "complementary approach" — you could even combine your model with their augmentation in Future Work.

### 15. "Tracing Like a Clinician" — Anatomy-Guided Spatial Priors
- **Venue:** arXiv (June 2026)
- **Method:** MobileNetV2 + U-Net, soft tissue profiling, 3-tier confidence system from heatmap spread
- **Relevance to you:** Interesting for the confidence/uncertainty angle. Cite if you add uncertainty estimation.

---

## The Full Comparison Table

| # | Paper | Year | Venue | Landmarks | Dataset Size | SDR@2mm | MRE (mm) | Method Type |
|---|---|---|---|---|---|---|---|---|
| 1 | CL-Detection 2024 Winner | 2024 | MICCAI | 53 | 700 | 82.0% | 1.19 | ConvNeXt+RCNN |
| 2 | D-CeLR | 2024 | ECCV | 19 | 400 | — | SOTA | Dual-encoder regression |
| 3 | CeLDA | 2024 | MICCAI | — | Custom | — | — | Prototypical network |
| 4 | YOLOv12 | 2026 | Sci. Reports | 19 | 499 | 80.6% | — | Bounding box |
| 5 | CBCT U-Net | 2026 | Sci. Reports | 3D | 75 | ~59%* | ≤1.0 | U-Net + attention |
| 6 | CephaNN | 2026 | MDPI | — | — | — | — | Dual U-Net ensemble |
| 7 | AIP U-Net | 2026 | AIP Conf. | 19 | — | — | — | U-Net + heatmap |
| 8 | CephRes-MHNet | 2025 | Withdrawn | **29** | **1000** | 85.5% | 1.23 | Multi-head residual |
| 9 | YOLOv11 | 2026 | Preprint | 4 | — | 87.2%** | 3.10 | YOLO detection |
| 10 | Diffusion augment | 2025 | arXiv | — | — | 82.2% | — | Diffusion + detection |
| **11** | **Your work (6 LM)** | **2026** | **—** | **6** | **1000** | **94.9%** | **0.78** | **U-Net + heatmap** |
| **12** | **Your work (29 LM)** | **2026** | **Target** | **29** | **1000** | **TBD** | **TBD** | **U-Net + heatmap** |

*\*Clinical interchangeability metric. \*\*SDR@4mm, not @2mm.*

---

## Which Papers You MUST Cite

| Paper | Why | Section |
|---|---|---|
| **Aariz dataset (Khalid 2025)** | Your dataset source | Methods |
| **YOLOv12 (2026)** | Primary comparison, same venue | Results, Discussion |
| **CephRes-MHNet (2025, withdrawn)** | Same dataset + landmarks, your benchmark to beat | Results |
| **CL-Detection 2024 winner** | Current MICCAI SOTA, establishes the performance ceiling | Introduction, Related Work |
| **D-CeLR (ECCV 2024)** | Alternative paradigm (regression vs heatmap) | Related Work |
| **CephaNN (MDPI 2026)** | Same method family (U-Net + heatmap) | Related Work |
| **Diffusion augmentation (2025)** | Complementary approach for Future Work | Related Work, Discussion |

---

## Where Your Work Sits

Your project is **not** competing against MICCAI challenge winners on 53-landmark multi-center datasets — that's a different tier. You're positioned in the **journal benchmark tier**, where:

- The bar is **CephRes-MHNet's 85.5% SDR@2mm / 1.23mm MRE** on Aariz (withdrawn)
- The published bar is **YOLOv12's 80.57% SDR@2mm** on a smaller dataset
- If your 29-landmark model achieves **SDR@2mm > 85% and MRE < 1.3mm**, you beat everything published on comparable datasets
- Your unique angles (pixel spacing calibration, multi-encoder ablation, clinical angle analysis) are things nobody else has done
