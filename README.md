# 🦷 AI-Powered Cephalometric Landmark Detection

**Automated orthodontic analysis using deep learning to detect anatomical landmarks on lateral skull X-rays and compute clinically meaningful jaw alignment measurements.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## Table of Contents

- [What Is This?](#what-is-this)
- [Why Does It Matter?](#why-does-it-matter)
- [What Does It Do?](#what-does-it-do)
- [Results & Performance](#results--performance)
- [Architecture & Technical Details](#architecture--technical-details)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset Download](#dataset-download)
- [Running the App](#running-the-app)
- [Training a Model](#training-a-model)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Course Techniques Used](#course-techniques-used)
- [Limitations & Future Improvements](#limitations--future-improvements)
- [Dataset](#dataset)
- [How It Works (Technical Deep-Dive)](#how-it-works-technical-deep-dive)
- [Acknowledgements](#acknowledgements)

---

## What Is This?

In orthodontics, a **cephalometric analysis** is the standard way to assess a patient's jaw alignment before planning treatments like braces, aligners, or jaw surgery. A dentist takes a **lateral (side-profile) skull X-ray** called a cephalogram, then manually marks specific anatomical points on the skull — things like the center of the pituitary fossa ("Sella"), the bridge of the nose ("Nasion"), and points along the jaw.

By drawing lines between these landmarks and measuring angles, orthodontists determine whether the upper jaw (maxilla) and lower jaw (mandible) are properly aligned, too far forward, or too far back.

**This project automates that entire process using AI.** Upload a skull X-ray, and the model:

1. **Detects 6 anatomical landmarks** on the image
2. **Draws the analysis lines** directly on the X-ray
3. **Computes three clinical angles** (SNA, SNB, ANB)
4. **Generates a plain-English interpretation** of what the angles mean — written so that anyone, not just a dentist, can understand the results

---

## Why Does It Matter?

| Problem | Our Solution |
|---------|-------------|
| Manual landmark placement takes **5–10 minutes** per X-ray | AI does it in **< 2 seconds** |
| Results vary between practitioners (inter-observer error ~0.5mm) | Consistent, reproducible predictions (MRE: 0.78mm) |
| Requires specialized orthodontic training to interpret | Plain-English interpretation accessible to anyone |
| Error-prone when fatigued (clinics process dozens daily) | No fatigue — same accuracy on image #1 and image #1000 |

> **Note:** This is a research/demo tool, not a certified medical device. Clinical decisions should always involve a qualified orthodontist.

---

## What Does It Do?


### The 6 Detected Landmarks

| # | Landmark | Symbol | What It Is (Plain English) |
|---|----------|--------|---------------------------|
| 1 | **Sella** | S | Center of the bony cavity that holds the pituitary gland, deep inside the skull |
| 2 | **Nasion** | N | The bridge of the nose — where the forehead bone meets the nasal bones |
| 3 | **A-point** | A | The deepest concavity on the front of the upper jaw bone |
| 4 | **B-point** | B | The deepest concavity on the front of the lower jaw bone |
| 5 | **Pogonion** | Pog | The most forward point of the chin |
| 6 | **Menton** | Me | The lowest point of the chin |

### The 3 Clinical Angles

| Angle | What It Measures | Normal Range | Interpretation |
|-------|-----------------|--------------|----------------|
| **SNA** | Upper jaw position relative to the skull base | 80–84° | High = upper jaw too far forward |
| **SNB** | Lower jaw position relative to the skull base | 78–82° | Low = receded chin |
| **ANB** | Relationship between upper and lower jaws (SNA − SNB) | 1–4° | Most important: determines Class I/II/III |

**ANB Classification:**
- **Class I** (ANB 1–4°): Normal jaw alignment ✅
- **Class II** (ANB > 4°): Upper jaw ahead of lower jaw — "overbite" tendency 🟡
- **Class III** (ANB < 0°): Lower jaw ahead of upper jaw — "underbite" tendency 🔴

---

## Results & Performance

Trained on the **Aariz dataset** (700 training images, 150 validation, 150 test) for 50 epochs on an Apple M4 GPU.

### Test Set Evaluation

| Metric | Value |
|--------|-------|
| **Overall MRE** | **0.78 mm** |
| **SDR @ 2.0mm** | **94.9%** |
| **SDR @ 2.5mm** | 97.4% |
| **SDR @ 3.0mm** | 98.8% |
| **SDR @ 4.0mm** | 99.7% |

> **MRE** (Mean Radial Error) = average distance between predicted and actual landmark positions.
> **SDR** (Successful Detection Rate) = percentage of landmarks within a given distance threshold.
> The **clinical acceptability threshold is 2mm** — our model places **94.9% of landmarks within this tolerance**.

### Per-Landmark Accuracy

| Landmark | MRE (mm) |
|----------|----------|
| Menton (Me) | 0.51 |
| Pogonion (Pog) | 0.61 |
| Sella (S) | 0.73 |
| B-point (B) | 0.92 |
| Nasion (N) | 0.94 |
| A-point (A) | 0.98 |

### Context

| Method | Overall MRE | SDR @ 2mm |
|--------|------------|-----------|
| Human inter-observer variability | 0.49 mm | — |
| **This model (U-Net + ResNet34)** | **0.78 mm** | **94.9%** |
| Literature SOTA (multi-stage) | ~1.0 mm | 80–90% |

Our single-stage model achieves **near-expert-level precision**.

---

## Architecture & Technical Details

### Model Architecture

```
Input (1×512×512 grayscale) 
    → ResNet34 Encoder (pretrained on ImageNet)
    → U-Net Decoder (5 upsampling blocks)
    → 6-channel Heatmap Output (one per landmark)
    → Weighted Centroid Refinement
    → (x, y) Coordinates
```

- **Backbone:** ResNet34 encoder from [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- **Architecture:** U-Net with skip connections for precise spatial localization
- **Output:** 6 Gaussian heatmaps (512×512), one per landmark
- **Post-processing:** Argmax → weighted centroid refinement for sub-pixel accuracy
- **Loss:** Mean Squared Error (MSE) on heatmap regression
- **Optimizer:** Adam with differential learning rates (encoder: 1e-4, decoder: 1e-3)
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Hardware:** Apple M4 MPS (Metal Performance Shaders) GPU acceleration

### Training Pipeline

```
Raw X-ray Images + JSON Annotations
    → Dual-annotator averaging (junior + senior orthodontist)
    → Per-image pixel spacing lookup (from CSV)
    → Gaussian heatmap generation (σ=5.0)
    → Keypoint-aware augmentation (Albumentations)
    → U-Net training with MSE loss
    → Checkpoint saving (best val loss)
    → Test evaluation (MRE in mm, SDR at 2/2.5/3/4mm)
```

### Evaluation Pipeline

- **MRE (Mean Radial Error):** Euclidean distance in **millimeters** (not pixels), using per-image pixel spacing from the original machine metadata
- **SDR (Successful Detection Rate):** Percentage of landmarks within 2.0mm, 2.5mm, 3.0mm, and 4.0mm thresholds
- **Per-landmark breakdown:** Individual MRE for each of the 6 landmarks

---

## Project Structure

```
cephalometry/
├── app/
│   └── streamlit_app.py          # Streamlit web interface (inference + visualization)
├── src/
│   ├── data/
│   │   ├── dataset.py             # PyTorch Dataset: JSON parsing, heatmap generation
│   │   └── preprocessing.py       # Image preprocessing utilities
│   ├── models/
│   │   └── unet.py                # U-Net factory (segmentation-models-pytorch)
│   ├── training/
│   │   ├── trainer.py             # Training loop with MPS support
│   │   └── mlflow_utils.py        # MLflow experiment tracking helpers
│   ├── inference/
│   │   ├── predict.py             # Heatmap → coordinates + confidence
│   │   └── angles.py              # SNA/SNB/ANB angle computation + interpretation
│   ├── viz/
│   │   └── overlay.py             # Matplotlib-based landmark visualization
│   ├── evaluation.py              # MRE/SDR evaluation with per-image pixel spacing
│   └── data_download.py           # Automatic dataset fetcher (from Figshare)
├── train.py                       # Training entry point
├── run.sh                         # Unified CLI (setup, train, docker, mlflow, local)
├── Dockerfile                     # Production container (CPU-only inference)
├── .dockerignore                  # Docker build context exclusions
├── .streamlit/                    # Streamlit theme configuration
├── requirements.txt               # Python dependencies (full, with GPU support)
├── requirements-docker.txt        # Python dependencies (CPU-only, for Docker)
├── checkpoints/                   # Saved model weights (best_model.pth)
├── sample_images/                 # Demo X-rays for the Streamlit app
└── data/                          # Training data (not committed — auto-downloaded)
    ├── cephalogram_machine_mappings.csv
    ├── train/
    │   ├── Cephalograms/          # X-ray images (.png/.jpg/.bmp)
    │   └── Annotations/           # JSON landmark annotations
    ├── valid/
    └── test/
```

---

## Setup & Installation

### Prerequisites

- **Python 3.12** (or compatible)
- **Docker** (for containerized deployment)
- macOS, Linux, or Windows with Python support

### Key Dependencies

| Package | Purpose |
|---------|---------|
| **PyTorch** ≥ 2.0 | Deep learning framework (model training and inference) |
| **segmentation-models-pytorch** ≥ 0.3 | Pretrained U-Net architecture with ResNet34 encoder |
| **Streamlit** ≥ 1.30 | Web UI for the prediction demo app |
| **MLflow** ≥ 2.10 | Experiment tracking (hyperparameters, metrics, artifacts) |
| **Albumentations** ≥ 1.3 | Keypoint-aware image augmentation during training |
| **OpenCV** (headless) ≥ 4.9 | Image I/O and processing |
| **NumPy**, **Pandas** | Array operations and CSV/metadata handling |
| **Matplotlib** | Landmark overlay visualization |
| **Pillow** ≥ 10.0 | Image loading and drawing in the Streamlit app |
| **scikit-learn** ≥ 1.3 | Utility functions for evaluation |

All dependencies are installed automatically via `pip install -r requirements.txt`.

### Option A: Using `run.sh` (Recommended)

```bash
git clone https://github.com/thegreatone9/CephalometryLandmarkDetection.git
cd CephalometryLandmarkDetection

./run.sh setup
```

This creates a `.venv` virtual environment and installs all dependencies.

### Option B: Manual virtual environment

```bash
git clone https://github.com/thegreatone9/CephalometryLandmarkDetection.git
cd CephalometryLandmarkDetection

python3.12 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Download

The Aariz dataset (~2 GB) is hosted on [Figshare](https://doi.org/10.6084/m9.figshare.27986417.v1) and is **not** included in the repository. Use the provided download script to fetch it automatically:

### Option A: Using `run.sh`

```bash
./run.sh setup       # if not already done
.venv/bin/python src/data_download.py
```

### Option B: Direct

```bash
source .venv/bin/activate
python src/data_download.py
```

The script will:
1. Download the dataset zip from Figshare (~2 GB)
2. Extract it into `data/`
3. Flatten any nested directories
4. Validate the expected structure (`train/`, `valid/`, `test/` with `Cephalograms/` and `Annotations/`)

After completion, the `data/` directory will contain:

```
data/
├── cephalogram_machine_mappings.csv
├── train/
│   ├── Cephalograms/                          # 700 X-ray images
│   └── Annotations/
│       └── Cephalometric Landmarks/
│           ├── Junior Orthodontists/           # JSON annotations
│           └── Senior Orthodontists/           # JSON annotations
├── valid/                                      # 150 images (same structure)
└── test/                                       # 150 images (same structure)
```

---

## Running the App

### Option A: Docker (Recommended for Demo)

Using `run.sh`:

```bash
./run.sh docker build      # Build the image (~6 min)
./run.sh docker start      # Start container → http://localhost:8501
./run.sh docker stop       # Stop container
```

Using bare Docker commands:

```bash
docker build -t cephalometric-demo:0.1 .
docker run -d --name cephalometric-demo -p 8501:8501 cephalometric-demo:0.1

# Open http://localhost:8501

docker stop cephalometric-demo
docker rm cephalometric-demo
```

The Docker image uses CPU-only PyTorch (`requirements-docker.txt`) to keep the image small (~2 GB). No GPU required for inference.

### Option B: Local with `run.sh`

```bash
./run.sh local start       # Start Streamlit → http://localhost:8501
./run.sh local stop        # Stop Streamlit
```

### Option C: Local with bare commands

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py --server.port=8501 --server.headless=true
```

### Using the App

1. Open **http://localhost:8501** in a browser
2. The app auto-detects the trained model checkpoint
3. Either **upload a cephalogram** (lateral skull X-ray) or click **"Load Sample"**
4. The model will:
   - Detect all 6 landmarks and draw them on the X-ray with color-coded markers
   - Display SNA, SNB, and ANB angle measurements
   - Generate a detailed clinical interpretation in plain English
   - Show per-landmark confidence scores

---

## Training a Model

### 1. Download the Data

```bash
python src/data_download.py       # See "Dataset Download" section
```

### 2. Run Training

Using `run.sh`:

```bash
# Quick smoke test (2 epochs)
./run.sh train --epochs 2 --batch-size 4

# Full training (50 epochs, ~50 min on M4 MPS)
./run.sh train --epochs 50 --batch-size 4

# Custom configuration
./run.sh train --epochs 100 --batch-size 8 --img-size 512 --encoder resnet34
```

Using bare commands:

```bash
source .venv/bin/activate
python train.py --epochs 50 --batch-size 4 --encoder resnet34 --img-size 512
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 4 | Batch size (4 is safe for M4 16GB) |
| `--img-size` | 512 | Input image resolution |
| `--encoder` | resnet34 | Encoder backbone |
| `--encoder-lr` | 1e-4 | Encoder learning rate |
| `--decoder-lr` | 1e-3 | Decoder learning rate |
| `--sigma` | 5.0 | Gaussian heatmap sigma |
| `--data-dir` | data | Path to dataset |
| `--checkpoint-dir` | `checkpoints/<run-name>` | Directory to save model weights (auto-named per run) |

### Hardware Acceleration

The training script auto-detects available hardware:

| Platform | Device | Expected Speed |
|----------|--------|---------------|
| Apple Silicon (M1/M2/M3/M4) | MPS | ~60s/epoch |
| NVIDIA GPU | CUDA | ~20–40s/epoch |
| CPU only | CPU | ~5–10min/epoch |

---

## Experiment Tracking with MLflow

All training runs are automatically logged to MLflow.

### Starting the MLflow UI

Using `run.sh`:

```bash
./run.sh mlflow start      # Start MLflow UI → http://localhost:5000
./run.sh mlflow stop       # Stop MLflow UI
```

Using bare commands:

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000
# Open http://localhost:5000
```

### What Gets Logged

| Category | Metrics |
|----------|---------|
| **Hyperparameters** | encoder, epochs, batch_size, learning rates, sigma, image size |
| **Per-epoch** | train_loss, val_loss |
| **Evaluation** | mre_overall_mm, per-landmark MRE, SDR at 2/2.5/3/4mm |
| **Artifacts** | Best model checkpoint (best_model.pth) |
| **Tags** | encoder name, run name (e.g. `resnet34-ep50-bs4-img512`) |

### MLflow Run Names

Runs are named descriptively as `{encoder}-ep{epochs}-bs{batch}-img{size}`, e.g. `resnet34-ep50-bs4-img512`, making it easy to compare experiments.

### Storage

MLflow uses a local SQLite database (`mlflow.db`) for metric storage. This file is gitignored.

---

## Course Techniques Used

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **Transfer Learning** | `src/models/unet.py` | ResNet34 encoder pretrained on ImageNet, fine-tuned on cephalometric X-rays with differential learning rates |
| **Data Preprocessing** | `src/data/dataset.py` | Grayscale normalization, resize to 512×512, dual-annotator averaging, Gaussian heatmap target generation |
| **Data Augmentation** | `src/data/preprocessing.py` | Albumentations pipeline with keypoint-aware rotation (±15°), brightness/contrast, Gaussian noise |
| **Model Training** | `src/training/trainer.py` | MSE loss, Adam optimizer with differential LR, ReduceLROnPlateau scheduler, best-model checkpointing |
| **Hyperparameter Tuning** | `train.py` | Configurable encoder, learning rates, sigma, image size, batch size — compared across MLflow runs |
| **MLflow Experiment Tracking** | `src/training/mlflow_utils.py` | Logs hyperparameters, per-epoch loss curves, evaluation metrics, and model artifacts |
| **Model Evaluation** | `src/evaluation.py` | MRE in millimeters (using per-image pixel spacing), SDR at multiple thresholds |
| **Dockerization** | `Dockerfile` | CPU-only production container serving the Streamlit prediction app |
| **API/UI Serving** | `app/streamlit_app.py` | Interactive web UI with image upload, real-time inference, clinical interpretation |
| **Reproducible Project Structure** | `run.sh`, `.gitignore`, `requirements.txt` | Automated setup, data fetching, training, deployment via single CLI script |

---

## Limitations & Future Improvements

### Current Limitations

- **6 of 29 landmarks:** The model only detects 6 of the 29 available landmarks in the Aariz dataset. A full cephalometric analysis (Steiner, Ricketts, McNamara) requires additional landmarks including dental points and soft-tissue markers.
- **Single-stage architecture:** The model uses a single U-Net pass. Multi-stage approaches (coarse detection → fine refinement) may improve precision on harder landmarks like A-point and Nasion.
- **No soft-tissue analysis:** The current model only detects skeletal (hard-tissue) landmarks. Soft-tissue landmarks (lip, nose tip, chin pad) are annotated in the Aariz dataset but not used.
- **Image quality sensitivity:** The model was trained on clinical-grade X-rays. Performance on low-quality, rotated, or non-standard cephalograms has not been evaluated.
- **Limited angle analysis:** Only SNA, SNB, and ANB angles are computed. Clinical practice uses many additional measurements (Wits appraisal, FMA, IMPA, nasolabial angle, etc.).
- **No CVM staging:** The Aariz dataset includes Cervical Vertebral Maturation (CVM) stage labels, which are important for growth prediction. This project does not use them.
- **Single dataset:** The model was trained and evaluated only on the Aariz dataset. Cross-dataset generalization (e.g., to the ISBI 2015 challenge dataset) has not been tested.

### Future Improvements

- **Expand to all 29 landmarks** for complete Steiner, Ricketts, and McNamara analyses
- **Add a refinement stage** (patch-based second network around each predicted landmark) to improve sub-pixel accuracy on difficult landmarks
- **Include soft-tissue landmarks** for profile analysis and surgical planning
- **Add CVM stage classification** as a multi-task output head on the same U-Net backbone
- **Cross-dataset evaluation** against the ISBI 2015 Cephalometric Landmark Detection Challenge dataset
- **Ensemble models** — average predictions from multiple architectures (ResNet34, EfficientNet-B4, HRNet) for more robust detection
- **Uncertainty estimation** using Monte Carlo dropout or deep ensembles to provide calibrated confidence intervals

---

## Dataset

This project uses the **Aariz dataset**, a large-scale cephalometric landmark detection dataset published in Nature Scientific Data:

> **Citation:** Khalid, M. A., et al. "Aariz: A Benchmark Dataset for Automatic Cephalometric Landmark Detection and CVM Stage Classification." *Scientific Data* 12, 1336 (2025). [https://doi.org/10.1038/s41597-025-05542-3](https://doi.org/10.1038/s41597-025-05542-3)

**Download:** [Figshare — doi:10.6084/m9.figshare.27986417.v1](https://doi.org/10.6084/m9.figshare.27986417.v1)

### Dataset Details

| Property | Value |
|----------|-------|
| **Total images** | 1,000 lateral cephalograms |
| **Split** | 700 train / 150 val / 150 test |
| **Landmarks annotated** | 29 (we use 6) |
| **Annotators** | Junior + Senior orthodontists |
| **Image format** | PNG, JPG, BMP (from 7 imaging devices) |
| **Pixel spacing** | 5 unique values (0.089–0.144 mm/px) |
| **Annotation format** | JSON with symbol-based lookup |

### Annotation Processing

The dataset provides annotations from two expert groups (junior and senior orthodontists). Our pipeline:

1. Parses JSON annotation files with symbol-based landmark lookup (e.g., "S" for Sella)
2. **Averages coordinates** from both annotator groups for more robust ground truth
3. Loads per-image **pixel spacing** from `cephalogram_machine_mappings.csv` for millimeter-scale evaluation
4. Generates 2D **Gaussian heatmaps** (σ=5.0) as regression targets

---

## How It Works (Technical Deep-Dive)

### 1. Data Loading (`src/data/dataset.py`)

The custom PyTorch `Dataset` class handles the Aariz format:

- Parses nested JSON annotations with symbol-based landmark lookup
- Averages coordinates from junior and senior annotator groups
- Generates 6-channel Gaussian heatmap targets (one channel per landmark)
- Applies keypoint-aware spatial augmentation (rotation, brightness, noise) using Albumentations with `KeypointParams`
- Loads images as single-channel grayscale, resized to 512×512

### 2. Model (`src/models/unet.py`)

- Uses `segmentation_models_pytorch.Unet` with a **ResNet34 encoder** pretrained on ImageNet
- Modified for **1-channel input** (grayscale X-rays) and **6-channel output** (one heatmap per landmark)
- Total parameters: ~24M

### 3. Training (`src/training/trainer.py`)

- **Loss:** MSE between predicted and ground-truth heatmaps
- **Optimizer:** Adam with differential learning rates — encoder (pretrained) gets a lower LR (1e-4) than the decoder (1e-3)
- **Scheduler:** ReduceLROnPlateau monitors validation loss with patience=5
- **Device:** Auto-detects MPS (Apple Silicon), CUDA (NVIDIA), or CPU
- **Checkpointing:** Saves best model (by validation loss) to `checkpoints/best_model.pth`

### 4. Post-Processing (`src/inference/predict.py`)

Converts raw heatmap output to precise coordinates:

1. **Argmax:** Find the peak pixel in each heatmap channel
2. **Weighted centroid refinement:** Extract a local patch around the peak and compute the intensity-weighted center of mass for **sub-pixel accuracy**
3. **Confidence scoring:** Peak heatmap value normalized to 0–100%

### 5. Angle Computation (`src/inference/angles.py`)

From the 6 detected landmarks, computes:

- **SNA** = angle at Nasion between Sella and A-point (upper jaw position)
- **SNB** = angle at Nasion between Sella and B-point (lower jaw position)
- **ANB** = SNA − SNB (jaw relationship — the key diagnostic metric)

Uses `atan2`-based angle computation for numerical stability.

### 6. Evaluation (`src/evaluation.py`)

- Computes **MRE in millimeters** (not pixels) using per-image pixel spacing from the dataset metadata
- Handles varying pixel spacings across different X-ray machines (5 unique values: 0.089, 0.1, 0.135, 0.139, 0.144 mm/px)
- Calculates **SDR at 4 thresholds** (2.0, 2.5, 3.0, 4.0mm)

### 7. Visualization (`app/streamlit_app.py`)

The Streamlit app provides:

- **Image upload** or sample image loading
- **Landmark overlay** with color-coded markers and labels
- **Angle measurement display** with delta from normal
- **Clinical interpretation** in plain English covering all 3 angles, with severity indicators
- **Confidence score bars** with warnings for low-confidence predictions
- **Disclaimer** for responsible use

---

## CLI Reference

The `run.sh` script is the unified entry point for all operations:

```bash
./run.sh setup              # Create venv and install dependencies
./run.sh train [OPTIONS]    # Train the model
./run.sh docker build       # Build the Docker image
./run.sh docker start       # Start the Docker container
./run.sh docker stop        # Stop the Docker container
./run.sh local start        # Start Streamlit locally
./run.sh local stop         # Stop local Streamlit
./run.sh mlflow start       # Start MLflow UI (port 5000)
./run.sh mlflow stop        # Stop MLflow UI
./run.sh help               # Show all available commands
```

---

## Acknowledgements

- **Dataset:** [Aariz — A Benchmark Dataset for Automatic Cephalometric Landmark Detection](https://www.nature.com/articles/s41597-025-05542-3) (Nature Scientific Data, 2025)
- **Model backbone:** [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) by Pavel Iakubovskii
- **Experiment tracking:** [MLflow](https://mlflow.org/)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **GPU acceleration:** Apple Metal Performance Shaders (MPS)
