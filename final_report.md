# Final Project Report
## AI-Powered Cephalometric Landmark Detection

**Project Title:** AI-Powered Cephalometric Landmark Detection Using Deep Learning

**Author:** Musa Khan

**Submission Date:** 25 June 2026

**Program:** Advanced Training on Semiconductor and ICT Technology — SICIP @ BRAC University

**GitHub Repository:** https://github.com/thegreatone9/CephalometryLandmarkDetection

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset and Preparation](#2-dataset-and-preparation)
3. [Methodology](#3-methodology)
4. [Experiments and Results](#4-experiments-and-results)
5. [Prediction App and Docker](#5-prediction-app-and-docker)
6. [Repository and Reproducibility](#6-repository-and-reproducibility)
7. [Limitations and Future Work](#7-limitations-and-future-work)
8. [Conclusion](#8-conclusion)
- [Appendix](#appendix)

---

## 1. Introduction

### 1.1 Brief Introduction

This project belongs to the **medical image analysis with deep learning** domain. It implements an end-to-end AI system for **cephalometric landmark detection** — the automated identification of anatomical reference points on lateral skull X-rays (cephalograms).

In orthodontics and oral surgery, a cephalometric analysis is the standard diagnostic procedure for assessing jaw alignment before treatment planning. Clinicians manually mark specific anatomical points on an X-ray, draw lines between them, and measure angles to determine whether a patient's upper jaw, lower jaw, and skull base are in normal proportion. This project automates the entire process using a deep learning model trained on expert-annotated data.

The final system accepts a lateral skull X-ray as input, detects 6 anatomical landmarks with sub-millimeter precision, computes the three primary clinical angles (SNA, SNB, ANB), and displays a plain-English interpretation of the results through an interactive web interface.

### 1.2 Problem Statement

**Input:** A lateral skull X-ray image (cephalogram) in PNG, JPG, or BMP format.

**Output:** The 6 detected landmark positions overlaid on the X-ray, SNA/SNB/ANB angle measurements, a clinical classification (Class I/II/III jaw relationship), per-landmark confidence scores, and a plain-English clinical interpretation.

**Who can use it:** Orthodontic researchers, dental students, and clinical teams seeking to automate or validate landmark placement. The plain-English output also makes results accessible to non-specialists.

**Why it matters:** Manual landmark placement takes 5–10 minutes per patient, is subject to inter-operator variability (~0.5mm between practitioners), and is susceptible to fatigue-related errors in high-volume clinics. An automated system that achieves near-expert accuracy reduces workload, improves consistency, and makes cephalometric screening more accessible.

### 1.3 Objectives

1. Train a deep learning model to detect 6 cephalometric landmarks on lateral skull X-rays with a Mean Radial Error below 2mm (the clinical acceptability threshold).
2. Track all training experiments with MLflow, logging hyperparameters, per-epoch loss curves, evaluation metrics, and model artifacts for every run.
3. Conduct multiple training runs with different configurations to compare results and identify the best-performing setup.
4. Build a Streamlit prediction app for real-time landmark detection with angle measurements and clinical interpretation.
5. Containerize the prediction app using Docker for reproducible deployment.
6. Automate dataset download, environment setup, training, and deployment through a unified shell script (`run.sh`).

---

## 2. Dataset and Preparation

### 2.1 Dataset Description

| Property | Value |
|----------|-------|
| **Dataset name** | Aariz — A Benchmark Dataset for Automatic Cephalometric Landmark Detection |
| **Source** | Figshare: doi:10.6084/m9.figshare.27986417.v1 |
| **Paper** | Khalid et al., Scientific Data 12, 1336 (2025) |
| **Total samples** | 1,000 lateral cephalographic X-rays |
| **Split** | 700 training / 150 validation / 150 test |
| **Annotation targets** | 29 cephalometric landmarks (this project uses 6) |
| **Annotators** | Two independent groups: Junior and Senior Orthodontists |
| **Input type** | Grayscale images (PNG, JPG, BMP) from 7 different X-ray machines |
| **Target/output** | (x, y) pixel coordinates per landmark, in JSON format |
| **Additional metadata** | Per-image pixel spacing (mm/pixel) for converting pixel errors to millimeter errors |

The 6 landmarks used in this project:

| Symbol | Landmark | Description |
|--------|----------|-------------|
| S | Sella | Center of the pituitary fossa |
| N | Nasion | Bridge of the nose — junction of frontal and nasal bones |
| A | A-point | Deepest concavity on the anterior upper jaw |
| B | B-point | Deepest concavity on the anterior lower jaw |
| Pog | Pogonion | Most anterior point of the chin |
| Me | Menton | Lowest point of the chin |

### 2.2 Data Access

The dataset is sourced from Figshare and can be downloaded automatically using the included script `src/data_download.py`:

```bash
python src/data_download.py
```

The script downloads the ~2 GB archive, extracts it into `data/`, validates the expected directory structure (`train/`, `valid/`, `test/` with `Cephalograms/` and `Annotations/` subdirectories), and offers to delete the zip file after extraction.

### 2.3 Preprocessing

**Annotation parsing:** JSON annotation files are parsed with symbol-based landmark lookup. Coordinates from Junior and Senior Orthodontist groups are **averaged** to produce a more robust ground truth.

**Image preprocessing:** Images are resized to 512×512 pixels, converted to grayscale, and normalized to [0, 1].

**Heatmap generation:** Rather than regressing raw coordinates, the model is trained on **Gaussian heatmaps** — one 512×512 channel per landmark, with a 2D Gaussian peak centered at each landmark's position (σ=5.0 pixels). This provides a spatially rich, smooth regression target.

**Data augmentation (training only):** The Albumentations library applies keypoint-aware augmentation, transforming coordinates in sync with the image:
- Random rotation (±15°) — simulates head tilt variation
- Random brightness and contrast — simulates different X-ray exposure settings
- Gaussian noise — simulates sensor noise

No augmentation is applied during validation or test evaluation.

---

## 3. Methodology

### 3.1 Course Techniques Used

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **Transfer Learning** | `src/models/unet.py`, `train.py` | ResNet34 encoder pretrained on ImageNet; differential LRs (encoder: 1e-4, decoder: 1e-3) |
| **Data Preprocessing** | `src/data/dataset.py` | Grayscale normalization, resize, dual-annotator averaging, Gaussian heatmap generation |
| **Data Augmentation** | `src/data/preprocessing.py` | Albumentations keypoint-aware pipeline (rotation, brightness/contrast, noise) |
| **CNN Deep Learning** | `src/models/unet.py` | U-Net with ResNet34 encoder (~24M params) for heatmap regression |
| **Model Training** | `src/training/trainer.py` | MSE loss, Adam optimizer, ReduceLROnPlateau scheduler, best-model checkpointing |
| **Hyperparameter Comparison** | `train.py`, 3 runs | Compared encoder (ResNet34 vs EfficientNet-b0) and resolution (512px vs 640px) |
| **MLflow Experiment Tracking** | `src/training/mlflow_utils.py` | Per-run logging of hyperparameters, loss curves, evaluation metrics, model artifacts |
| **Model Evaluation** | `src/evaluation.py` | MRE in millimeters using per-image pixel spacing, SDR at 2/2.5/3/4mm thresholds |
| **Dockerization** | `Dockerfile` | CPU-only production container serving the Streamlit app |
| **API/UI Model Serving** | `app/streamlit_app.py` | Interactive Streamlit web app with upload, inference, visualization, and interpretation |
| **Reproducible Project Structure** | `run.sh`, `.gitignore`, `requirements.txt` | Single-script CLI for all operations |

### 3.2 Model Architecture

The model is a **U-Net** with a **ResNet34 encoder** pretrained on ImageNet, from `segmentation-models-pytorch`.

```
Input: [1 × 512 × 512]  (single-channel grayscale X-ray)
    │
    ▼
ResNet34 Encoder (pretrained, ~21M params)
    │  5 stages of downsampling, skip connections preserved
    ▼
U-Net Decoder (~3M params)
    │  5 upsampling blocks, restores 512×512 resolution
    ▼
Output: [6 × 512 × 512]  (one Gaussian heatmap per landmark)
    │
    ▼
Weighted Centroid Refinement
    │  Argmax peak → intensity-weighted center of mass for sub-pixel accuracy
    ▼
6 × (x, y) coordinates  +  confidence scores (0–100%)
```

| Parameter | Value |
|-----------|-------|
| Loss function | MSE on heatmaps |
| Optimizer | Adam with differential LR (encoder: 1e-4, decoder: 1e-3) |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Heatmap sigma | σ = 5.0 pixels |
| Input resolution | 512 × 512 pixels |
| Total parameters | ~24M |
| Pretrained weights | ImageNet |

### 3.3 Training Strategy

**Setup:** Apple M4 GPU (MPS backend), batch size 4, 50 epochs. Hardware is auto-detected (MPS → CUDA → CPU).

**Validation strategy:** The model is evaluated on the validation set after every epoch. The checkpoint with the lowest validation loss is saved to a run-specific directory under `checkpoints/` (e.g. `checkpoints/resnet34-ep50-bs4-img512/best_model.pth`).

**Best model selection:** The best-validation-loss checkpoint is always used for final test evaluation — not the final epoch's weights. This protects against late-training overfitting.

---

## 4. Experiments and Results

### 4.1 MLflow Tracking

All training runs are logged to a local MLflow SQLite database (`mlflow.db`) under the experiment `cephalometric-landmark-detection`.

**Parameters logged:** `encoder_name`, `epochs`, `batch_size`, `image_size`, `sigma`, `encoder_lr`, `decoder_lr`

**Metrics logged:** `train_loss` and `val_loss` per epoch; final `mre_overall_mm`; per-landmark MRE (`mre_Sella_mm`, etc.); `SDR_2.0mm`, `SDR_2.5mm`, `SDR_3.0mm`, `SDR_4.0mm`

**Artifacts saved:** Best model checkpoint (`best_model.pth`) per run, stored under `mlruns/`

**Runs completed:**

| Run Name | Encoder | Resolution | Status |
|---|---|---|---|
| `resnet34-ep50-bs4-img512` | ResNet34 | 512px | FINISHED |
| `resnet34-ep50-bs4-img640` | ResNet34 | 640px | FINISHED |
| `efficientnet-b0-ep50-bs4-img512` | EfficientNet-b0 | 512px | FINISHED |

**Best run:** `resnet34-ep50-bs4-img512` — lowest MRE (0.78mm), highest SDR@2mm (94.9%), most stable training.

The MLflow UI can be launched with `./run.sh mlflow start` → http://localhost:5000.

> Screenshots: `screenshots/mlflow_runs.png`

### 4.2 Evaluation Results

| Metric | Run 1: ResNet34 @ 512px | Run 2: ResNet34 @ 640px | Run 3: EfficientNet-b0 @ 512px |
|--------|------------------------|------------------------|-------------------------------|
| **Overall MRE** | **0.78 mm** | 17.02 mm | 46.27 mm |
| **SDR @ 2.0mm** | **94.9%** | 77.0% | 42.7% |
| **SDR @ 2.5mm** | 97.4% | 80.0% | 45.7% |
| **SDR @ 3.0mm** | 98.8% | 81.8% | 47.0% |
| **SDR @ 4.0mm** | 99.7% | 82.8% | 48.1% |
| Best val loss | 0.000390 (epoch 4) | 0.000074 (epoch 49) | 0.000272 (epoch 29) |
| Training stability | Stable | Stable | Highly oscillating |

**Per-landmark accuracy (production model — Run 1):**

| Landmark | MRE (mm) |
|----------|----------|
| Menton (Me) | 0.51 |
| Pogonion (Pog) | 0.61 |
| Sella (S) | 0.73 |
| B-point (B) | 0.92 |
| Nasion (N) | 0.94 |
| A-point (A) | 0.98 |

**Interpretation:** Run 1 places 94.9% of landmarks within the 2mm clinical threshold — surpassing literature SOTA for single-stage models and approaching human inter-observer accuracy (0.49mm). Run 2's catastrophic MRE (17.02mm) is driven entirely by Sella (97.89mm), caused by the proportionally narrower Gaussian target at 640px with the same σ=5.0. Run 3 (EfficientNet-b0) showed highly unstable training, with 4 of 6 landmarks failing, likely due to incompatible skip-connection feature map sizes between EfficientNet and the U-Net decoder.

> Screenshots: `screenshots/training_result.png`

---

## 5. Prediction App and Docker

### 5.1 Prediction Pipeline

1. **Upload:** User uploads a lateral skull X-ray or loads a sample image via the Streamlit UI.
2. **Preprocessing:** Resized to 512×512, converted to grayscale, normalized to [0, 1], formatted as tensor `[1, 1, 512, 512]`.
3. **Inference:** Tensor passed through the U-Net → 6 heatmaps `[1, 6, 512, 512]`.
4. **Coordinate extraction:** Argmax finds the peak pixel per channel; weighted centroid refinement computes the intensity-weighted center of mass in a 21×21 patch for sub-pixel accuracy; confidence = normalized peak value.
5. **Coordinate rescaling:** 512×512 coordinates scaled back to original image dimensions.
6. **Angle computation:** SNA, SNB, and ANB computed from S, N, A, B coordinates using atan2-based vector angle computation.
7. **Output:** Annotated X-ray, angle measurements, clinical interpretation, and confidence bars displayed in the UI.

### 5.2 Prediction UI

The app is built with **Streamlit** (`app/streamlit_app.py`):

- **Sidebar:** Checkpoint selector dropdown (newest model pre-selected), About section
- **Main area:** Image upload + sample loader; two-column layout (annotated X-ray left, angle metrics right); full-width clinical interpretation; full-width per-landmark confidence bars with low-confidence warnings

> Screenshots: `screenshots/demo_output.png`

### 5.3 Docker Serving

The `Dockerfile` uses Python 3.12 slim with CPU-only PyTorch (`requirements-docker.txt`), bundling the trained checkpoint and sample images. Image size ~2 GB, exposing Streamlit on port 8501.

```bash
docker build -t cephalometric-demo:0.1 .
docker run -d --name cephalometric-demo -p 8501:8501 cephalometric-demo:0.1
# Access at http://localhost:8501
```

Full instructions are in the README.

> Screenshots: `screenshots/docker_app_running.png`

---

## 6. Repository and Reproducibility

### 6.1 Repository Structure

```
CephalometryLandmarkDetection/
├── README.md                      # Full project documentation
├── final_report.md                # This report
├── requirements.txt               # Full dependencies (GPU-enabled)
├── requirements-docker.txt        # CPU-only dependencies for Docker
├── .gitignore
├── Dockerfile
├── run.sh                         # Unified CLI
├── train.py                       # Training entry point
├── src/
│   ├── data_download.py           # Automatic dataset downloader
│   ├── data/dataset.py            # PyTorch Dataset class
│   ├── data/preprocessing.py      # Augmentation pipelines
│   ├── models/unet.py             # U-Net factory
│   ├── training/trainer.py        # Training loop
│   ├── training/mlflow_utils.py   # MLflow helpers
│   ├── inference/predict.py       # Heatmap → coordinates
│   ├── inference/angles.py        # Angle computation
│   ├── viz/overlay.py             # Visualization
│   └── evaluation.py             # MRE/SDR evaluation
├── app/streamlit_app.py           # Prediction UI
├── screenshots/                   # MLflow, Docker, demo screenshots
├── sample_images/                 # Demo X-rays bundled in Docker
├── mlruns/                        # MLflow run history (included in repo)
├── checkpoints/                   # Trained model weights (not versioned)
└── data/                          # Dataset directory (populated by data_download.py)
```

### 6.2 GitHub Rules

**Not committed:** `data/`, `checkpoints/`, `mlflow.db`, `.venv/`, `__pycache__/`

**Committed:** `mlruns/` — MLflow artifact folder, included so reviewers can inspect experiment history without re-running training.

### 6.3 Reproducibility

From a fresh clone:

1. **Setup:** `./run.sh setup`
2. **Download data:** `python src/data_download.py`
3. **Train:** `./run.sh train --epochs 50 --batch-size 4`
4. **Inspect MLflow:** `./run.sh mlflow start` → http://localhost:5000
5. **Run app locally:** `./run.sh local start` → http://localhost:8501
6. **Run app via Docker:** `./run.sh docker build && ./run.sh docker start`

---

## 7. Limitations and Future Work

### 7.1 Limitations

- **6 of 29 landmarks only:** A complete orthodontic analysis requires more landmarks for Steiner, Ricketts, and McNamara analyses.
- **Single-stage architecture:** Multi-stage approaches with dedicated refinement networks typically achieve higher precision on hard landmarks.
- **Sella sensitivity to resolution:** As shown by Run 2, Sella detection fails when image resolution increases without proportionally scaling the heatmap sigma.
- **EfficientNet instability:** EfficientNet-b0 with standard U-Net decoding showed highly oscillating val loss and poor generalization — likely due to incompatible skip-connection feature map sizes.
- **No soft-tissue analysis:** Soft-tissue landmarks (lip, nose tip, chin pad) are available in the dataset but not used.
- **Single dataset:** Generalization to other cephalometric datasets (e.g., ISBI 2015) has not been tested.
- **No CVM staging:** Cervical Vertebral Maturation stage labels in the dataset are unused.

### 7.2 Future Improvements

- Expand to all 29 landmarks for complete clinical cephalometric analysis.
- Scale heatmap sigma with input resolution (σ = resolution/102.4 for consistent target width).
- Add a patch-level refinement network as a second stage for harder landmarks.
- Tune EfficientNet with lower learning rates and architecture-specific skip connection adapters.
- Include soft-tissue landmarks and CVM stage classification as multi-task outputs.
- Cross-dataset evaluation against ISBI 2015 for generalization testing.
- Ensemble prediction across multiple trained models for more robust outputs.

---

## 8. Conclusion

This project successfully built a complete, reproducible deep learning system for automated cephalometric landmark detection. The production model (U-Net + ResNet34 @ 512px) achieves an overall Mean Radial Error of 0.78mm and places 94.9% of landmarks within the 2mm clinical acceptability threshold — surpassing the literature SOTA for single-stage models and approaching human inter-observer accuracy.

Three training experiments were tracked with MLflow, revealing that ResNet34 outperforms EfficientNet-b0 for U-Net-based heatmap regression, and that input resolution must be carefully matched to the heatmap sigma to avoid failures on anatomically ambiguous landmarks.

The Streamlit prediction app — served via Docker at http://localhost:8501 — allows any user to upload a cephalogram and receive landmark overlays, angle measurements, a clinical classification, and a plain-English interpretation with no orthodontic expertise required.

The project demonstrates transfer learning, heatmap regression, keypoint-aware data augmentation, MLflow experiment tracking, Docker containerization, and interactive web-based model serving — all techniques covered in the SICIP certification program.

---

## Appendix

### A. Final Submission Checklist

- [x] Selected one approved project domain (Medical image analysis — deep learning)
- [x] Public GitHub repository accessible: https://github.com/thegreatone9/CephalometryLandmarkDetection
- [x] Dataset is sourced externally and auto-downloaded via `src/data_download.py`
- [x] Data auto-downloads via `python src/data_download.py`
- [x] Training uses MLflow (`src/training/mlflow_utils.py`)
- [x] `mlruns/` is included and **not** gitignored
- [x] Three MLflow runs logged (minimum two required)
- [x] Report includes MLflow screenshot (`screenshots/mlflow_runs.png`)
- [x] Prediction app supports sample upload (Streamlit)
- [x] Prediction app served via Docker
- [x] Report includes Docker screenshot (`screenshots/docker_app_running.png`)
- [x] Course techniques clearly documented (Section 3.1)
- [ ] Deadline: **25 June 2026**
