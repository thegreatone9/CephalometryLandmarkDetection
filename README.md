# 🦷 AI-Powered Cephalometric Landmark Detection

**Automated orthodontic analysis using deep learning to detect anatomical landmarks on lateral skull X-rays and compute clinically meaningful jaw alignment measurements.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## Table of Contents

- [Cephalometric Analysis: The Domain](#cephalometric-analysis-the-domain)
  - [What Is a Cephalogram?](#what-is-a-cephalogram)
  - [What Are Landmarks?](#what-are-landmarks)
  - [What Are the Angles and Why Do They Matter?](#what-are-the-angles-and-why-do-they-matter)
  - [The Manual Process and Why Automation Helps](#the-manual-process-and-why-automation-helps)
- [What This Project Does](#what-this-project-does)
- [Results & Performance](#results--performance)
- [Project Structure](#project-structure)
- [Code Architecture Deep-Dive](#code-architecture-deep-dive)
  - [Training Entry Point: train.py](#1-training-entry-point-trainpy)
  - [Dataset & Heatmap Generation: dataset.py](#2-dataset--heatmap-generation-datasetpy)
  - [Data Augmentation: preprocessing.py](#3-data-augmentation-preprocessingpy)
  - [Model Architecture: unet.py](#4-model-architecture-unetpy)
  - [Training Loop: trainer.py](#5-training-loop-trainerpy)
  - [MLflow Integration: mlflow_utils.py](#6-mlflow-integration-mlflow_utilspy)
  - [Post-Processing: predict.py](#7-post-processing-predictpy)
  - [Angle Computation: angles.py](#8-angle-computation-anglespy)
  - [Evaluation Pipeline: evaluation.py](#9-evaluation-pipeline-evaluationpy)
  - [Visualization: overlay.py](#10-visualization-overlaypy)
  - [Streamlit App: streamlit_app.py](#11-streamlit-app-streamlit_apppy)
- [How run.sh Works](#how-runsh-works)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Training a Model](#training-a-model)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Dataset](#dataset)
- [Acknowledgements](#acknowledgements)

---

## Cephalometric Analysis: The Domain

### What Is a Cephalogram?

A **cephalogram** (also called a lateral cephalometric radiograph) is a specific type of skull X-ray taken from the side. The patient stands with their head in a fixed position (using a device called a cephalostat) and an X-ray is taken in perfect lateral (side-on) orientation.

Unlike a regular dental X-ray that shows teeth, a cephalogram captures the **entire skull profile** — the bones of the face, the jaw joints, the nasal cavity, and even the vertebrae. This gives orthodontists a 2D "blueprint" of the patient's skeletal structure.

Cephalograms have been the gold standard in orthodontic diagnosis since the 1930s. Every patient who gets braces, jaw surgery, or orthodontic treatment typically has one or more cephalograms taken during their treatment planning.

### What Are Landmarks?

**Cephalometric landmarks** are specific, anatomically defined points on the skull that orthodontists identify on the X-ray. Each landmark corresponds to a precise skeletal feature — a bone edge, a cavity, an intersection of structures. There are roughly 29 standard landmarks used in full cephalometric analysis. This project focuses on the 6 most clinically important ones:

| # | Landmark | Symbol | Anatomical Definition | Why It Matters |
|---|----------|--------|-----------------------|----------------|
| 1 | **Sella** | S | The geometric center of the pituitary fossa (sella turcica) — a saddle-shaped depression in the sphenoid bone at the base of the skull | Reference point for the cranial base. Forms the starting point of the SN line (Sella-Nasion), which serves as the "horizontal reference" for all angle measurements |
| 2 | **Nasion** | N | The most anterior point of the frontonasal suture — where the frontal bone (forehead) meets the nasal bones (bridge of the nose) | The vertex (pivot point) for both SNA and SNB angles. Its position determines the "neutral axis" of the skull |
| 3 | **A-point** | A | The deepest midline concavity on the maxilla (upper jaw bone), between the anterior nasal spine and the upper incisor root tips | Represents the anterior limit of the upper jaw. If this point is far forward, the upper jaw is protruding |
| 4 | **B-point** | B | The deepest midline concavity on the mandible (lower jaw bone), between the chin prominence (pogonion) and the lower incisor root tips | Represents the anterior limit of the lower jaw. Used to assess lower jaw position |
| 5 | **Pogonion** | Pog | The most anterior (forward-projecting) point of the bony chin | Defines the prominence of the chin. Used in profile analysis |
| 6 | **Menton** | Me | The lowest point on the bony chin outline (mandibular symphysis) | Defines the vertical extent of the lower face. Used in vertical dimension analysis |

### What Are the Angles and Why Do They Matter?

Once the landmarks are identified, the orthodontist draws lines between them and measures angles. The three key angles are:

#### SNA Angle (Upper Jaw Position)

```
         S (Sella)
          \
           \   ← Sella-Nasion line (cranial base reference)
            \
             N (Nasion)  ← vertex of the angle
            /
           /   ← Nasion-A line
          /
         A (A-point)
```

**What it measures:** The angle at Nasion between lines drawn to Sella and A-point. This tells you how far forward or backward the **upper jaw** sits relative to the skull base.

- **Normal range:** 80–84°
- **High SNA (>84°):** Upper jaw is positioned too far forward (maxillary prognathism). The upper teeth stick out.
- **Low SNA (<80°):** Upper jaw is positioned too far back (maxillary retrognathism). The midface looks flat.

#### SNB Angle (Lower Jaw Position)

Same construction but with B-point instead of A-point:

- **Normal range:** 78–82°
- **High SNB (>82°):** Lower jaw is positioned too far forward — prominent chin.
- **Low SNB (<78°):** Lower jaw is positioned too far back — receded/weak chin.

#### ANB Angle (Jaw Relationship)

**ANB = SNA − SNB.** This is the most important single measurement in cephalometrics because it directly quantifies how the upper and lower jaws relate to each other.

- **Normal range:** 1–4° (ideally ~2°)
- **Class I (ANB 1–4°):** Normal jaw alignment. The upper and lower jaws are in harmony. ✅
- **Class II (ANB > 4°):** The upper jaw is too far ahead of the lower jaw — "overbite" tendency. The upper front teeth overlap the lower front teeth excessively. 🟡
- **Class III (ANB < 0°):** The lower jaw is ahead of the upper jaw — "underbite" tendency. The lower front teeth sit in front of the upper front teeth. 🔴

These classifications (I, II, III) directly drive treatment decisions:
- Class I typically needs only dental corrections (straightening teeth)
- Class II may need functional appliances, elastics, or jaw surgery to advance the lower jaw
- Class III may need reverse-pull headgear, elastics, or jaw surgery to set back the lower jaw

### The Manual Process and Why Automation Helps

Traditionally, an orthodontist:
1. Takes the cephalogram X-ray
2. Places it on a lightbox or loads it on screen
3. Manually identifies each landmark by visual inspection (5–10 minutes)
4. Draws construction lines between landmarks
5. Measures angles with a protractor or software tool
6. Interprets the measurements against normal ranges
7. Writes up the analysis for the treatment plan

**Problems with the manual process:**
- **Inter-observer variability:** Two orthodontists marking the same X-ray will place landmarks slightly differently (average error ~0.5mm). This introduces uncertainty.
- **Time-consuming:** Each analysis takes 5–10 minutes, and a busy clinic may need to process dozens per day.
- **Fatigue-prone:** After 20 analyses, accuracy decreases.
- **Requires specialist training:** Only trained orthodontists can reliably identify landmarks.

**What automation provides:**
- Consistent, reproducible predictions (no observer variability)
- Sub-2-second analysis per image
- No fatigue — same accuracy on image #1 and image #1000
- Accessible results even for non-specialists (via plain-English interpretation)

---

## What This Project Does

Upload a lateral skull X-ray, and the model:

1. **Detects 6 anatomical landmarks** on the image with sub-millimeter precision
2. **Draws the analysis lines** (S–N, N–A, N–B) directly on the X-ray
3. **Computes three clinical angles** (SNA, SNB, ANB)
4. **Generates a plain-English interpretation** of what the angles mean — written so that anyone, not just a dentist, can understand the results
5. **Reports confidence scores** for each landmark with warnings for low-confidence predictions

> **Note:** This is a research/demo tool, not a certified medical device. Clinical decisions should always involve a qualified orthodontist.

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

> **MRE** (Mean Radial Error) = average Euclidean distance between predicted and actual landmark positions, converted to millimeters using per-image pixel spacing metadata.
>
> **SDR** (Successful Detection Rate) = percentage of landmarks within a given distance threshold.
>
> The **clinical acceptability threshold is 2mm** — our model places **94.9% of landmarks within this tolerance**.

### Per-Landmark Accuracy

| Landmark | MRE (mm) | Notes |
|----------|----------|-------|
| Menton (Me) | 0.51 | Easiest — sharp bone edge |
| Pogonion (Pog) | 0.61 | Clear convexity |
| Sella (S) | 0.73 | Well-defined cavity |
| B-point (B) | 0.92 | Subtle concavity, harder |
| Nasion (N) | 0.94 | Overlapping structures |
| A-point (A) | 0.98 | Subtle concavity, hardest |

### Context

| Method | Overall MRE | SDR @ 2mm |
|--------|------------|-----------|
| Human inter-observer variability | 0.49 mm | — |
| **This model (U-Net + ResNet34)** | **0.78 mm** | **94.9%** |
| Literature SOTA (multi-stage) | ~1.0 mm | 80–90% |

Our single-stage model achieves **near-expert-level precision**.

---

## Project Structure

```
cephalometry/
├── app/
│   └── streamlit_app.py          # Web interface: upload → inference → visualization
├── src/
│   ├── data/
│   │   ├── dataset.py             # PyTorch Dataset: JSON parsing, heatmap generation
│   │   └── preprocessing.py       # Albumentations augmentation pipelines
│   ├── models/
│   │   └── unet.py                # U-Net factory (segmentation-models-pytorch wrapper)
│   ├── training/
│   │   ├── trainer.py             # Training loop: MSE loss, Adam, checkpointing
│   │   └── mlflow_utils.py        # MLflow experiment tracking helpers
│   ├── inference/
│   │   ├── predict.py             # Heatmap → coordinates + weighted centroid refinement
│   │   └── angles.py              # SNA/SNB/ANB computation + clinical interpretation
│   ├── viz/
│   │   └── overlay.py             # Matplotlib-based landmark visualization
│   └── evaluation.py              # MRE/SDR evaluation with per-image pixel spacing
├── train.py                       # CLI training entry point
├── run.sh                         # Unified operations script (338 lines)
├── Dockerfile                     # CPU-only inference container
├── .dockerignore                  # Excludes data/, .venv/, .git/ from Docker context
├── requirements.txt               # Full dependencies (with GPU support for training)
├── requirements-docker.txt        # CPU-only dependencies (for Docker inference)
├── .streamlit/                    # Streamlit theme configuration
├── checkpoints/                   # Saved model weights (best_model.pth, ~280MB)
├── sample_images/                 # Demo X-rays for the Streamlit app
└── data/                          # Training data (not committed)
```

### Inter-Module Dependency Graph

```
train.py (orchestrator)
├── src/data/dataset.py ──── CephalometricDataset, constants, load_pixel_spacings
│   └── src/data/preprocessing.py ──── Albumentations augmentation pipelines
├── src/models/unet.py ──── create_model(), get_parameter_groups()
├── src/training/trainer.py ──── Trainer class (training loop)
│   ├── src/models/unet.py
│   └── src/training/mlflow_utils.py ──── log_epoch_metrics(), log_model_artifact()
├── src/training/mlflow_utils.py ──── setup_experiment(), log_hyperparams()
└── src/evaluation.py ──── evaluate_model()
    └── src/inference/predict.py ──── heatmap_to_coordinates(), refine_coordinates()

app/streamlit_app.py (standalone inference)
├── src/models/unet.py ──── create_model()
├── src/inference/predict.py ──── heatmap_to_coordinates(), compute_confidence()
└── src/inference/angles.py ──── compute_sna(), compute_snb(), compute_anb(), interpret_anb()
```

---

## Code Architecture Deep-Dive

### The Training Data Flow

Before diving into individual files, here's the complete data flow from raw files to trained model:

```
Raw X-ray PNG + JSON annotations
    ↓ dataset.py: parse JSON, average junior/senior annotations
    ↓ dataset.py: generate 6-channel Gaussian heatmaps (σ=5)
    ↓ preprocessing.py: apply keypoint-aware augmentation
    ↓ DataLoader: batch into tensors [B, 1, 512, 512]
    ↓ unet.py: U-Net forward pass → predicted heatmaps [B, 6, 512, 512]
    ↓ trainer.py: MSE loss vs target heatmaps → backprop → update weights
    ↓ trainer.py: checkpoint best model by val loss
    ↓ predict.py: argmax → weighted centroid refinement → (x, y) coords
    ↓ evaluation.py: rescale to original resolution → MRE in mm + SDR
    ↓ mlflow_utils.py: log all metrics
    ↓ Best model saved to checkpoints/best_model.pth
```

### 1. Training Entry Point: `train.py`

**Purpose:** The CLI orchestrator that ties everything together.

**What happens when you run `./run.sh train --epochs 50`:**

1. **Parse arguments** via `parse_args()` — encoder, epochs, batch size, learning rates, sigma, image size, data/checkpoint directories.

2. **Load pixel spacings** from `data/cephalogram_machine_mappings.csv`. This CSV maps each X-ray image to the physical pixel size of the machine that captured it (in mm/pixel). Values range from 0.089 to 0.144 mm/px across 5 different X-ray machines. If the CSV isn't found, defaults to 0.1 mm/px.

3. **Build DataLoaders** via `build_dataloaders()` — creates three `CephalometricDataset` instances for `train/`, `val/`, and `test/` splits, wraps them in PyTorch `DataLoader`s. The train loader shuffles; val/test do not.

4. **Set up MLflow** — creates or retrieves the `cephalometric-landmark-detection` experiment, starts a run named like `resnet34-ep50-bs4-img512`, logs all hyperparameters.

5. **Train** — instantiates `Trainer`, calls `trainer.fit(train_loader, val_loader)`. This runs the full training loop and returns the path to the best checkpoint.

6. **Evaluate** — loads the best checkpoint, runs `evaluate_model()` on the test DataLoader, logs MRE and SDR metrics to MLflow, prints a formatted results table.

### 2. Dataset & Heatmap Generation: `dataset.py`

**Purpose:** The most complex module — handles the Aariz dataset's JSON format, dual-annotator averaging, and Gaussian heatmap target generation.

**Constants:**
- `ALL_LANDMARK_SYMBOLS`: All 29 landmarks in the Aariz dataset
- `SELECTED_SYMBOLS = ["S", "N", "A", "B", "Pog", "Me"]`: The 6 we use
- `NUM_LANDMARKS = 6`

**JSON annotation parsing:** The Aariz dataset stores annotations in JSON files with a nested structure:

```json
{
  "landmarks": [
    {"symbol": "S", "value": {"x": 234.5, "y": 189.2}},
    {"symbol": "N", "value": {"x": 312.1, "y": 201.8}},
    ...
  ]
}
```

The function `_load_landmarks_from_json()` parses this into a `{symbol: (x, y)}` dict.

**Dual-annotator averaging:** Each image has annotations from both a junior and senior orthodontist (in separate JSON files). The function `_average_annotations()` loads both and computes the element-wise mean of their coordinates. This produces more reliable ground truth than either annotator alone.

**Gaussian heatmap generation:** Instead of predicting raw (x, y) coordinates directly (which is numerically unstable), the model predicts **heatmaps** — one 512×512 image per landmark, where the target is a 2D Gaussian blob centered on the landmark's location:

```
_generate_gaussian_heatmap(height, width, cx, cy, sigma=5.0):
    # Creates a 2D Gaussian with peak=1.0 at (cx, cy)
    # σ=5.0 means the "hot spot" spans roughly 10 pixels
    # Values decay to near-zero beyond ~3σ (15 pixels)
    heatmap[y, x] = exp(-((x-cx)² + (y-cy)²) / (2σ²))
```

This approach is standard in keypoint detection because:
- It provides a smooth, differentiable loss landscape (vs. discrete coordinate regression)
- Small errors in peak position result in small loss values (graceful degradation)
- The model can express uncertainty through peak spread/height

**`CephalometricDataset.__getitem__()`** returns a dict with:
- `image`: tensor `[1, 512, 512]` — grayscale, normalized to [0,1]
- `heatmaps`: tensor `[6, 512, 512]` — 6-channel Gaussian targets
- `landmarks`: tensor `[6, 2]` — raw (x, y) coordinates (for evaluation)
- `meta`: dict with `image_path`, `ceph_id`, `original_size`, `resized_size`, `pixel_spacing`

### 3. Data Augmentation: `preprocessing.py`

**Purpose:** Defines augmentation pipelines using Albumentations with **keypoint-aware transforms** — when the image is rotated or shifted, the landmark coordinates are automatically transformed to match.

**Training augmentations:**
- `Resize(512, 512)` — standardize input size
- `Rotate(limit=±15°, p=0.5)` — simulate head tilt during X-ray capture
- `RandomBrightnessContrast(±0.15, p=0.5)` — simulate exposure variation
- `GaussNoise(std_range=(0.01, 0.05), p=0.3)` — simulate sensor noise

**Critically omitted: horizontal flip.** Unlike natural images where flipping is a free augmentation, cephalograms are **always** taken from the same side. Flipping would create anatomically impossible images and confuse the model.

**Keypoint tracking:** Uses `KeypointParams(format="xy", remove_invisible=False)` so Albumentations transforms the (x, y) coordinates along with the image. `remove_invisible=False` ensures landmarks that rotate slightly outside the image boundary aren't dropped.

### 4. Model Architecture: `unet.py`

**Purpose:** Creates the U-Net model using `segmentation_models_pytorch`.

**Architecture:**

```
Input: [B, 1, 512, 512] (grayscale X-ray)
    │
    ▼ ResNet34 Encoder (pretrained on ImageNet)
    │  ├── Block 1: 64 channels,  256×256
    │  ├── Block 2: 64 channels,  128×128
    │  ├── Block 3: 128 channels, 64×64
    │  ├── Block 4: 256 channels, 32×32
    │  └── Block 5: 512 channels, 16×16  (bottleneck)
    │
    ▼ U-Net Decoder (5 upsampling blocks with skip connections)
    │  ├── Up 5: 256 channels, 32×32 + skip from Block 4
    │  ├── Up 4: 128 channels, 64×64 + skip from Block 3
    │  ├── Up 3: 64 channels,  128×128 + skip from Block 2
    │  ├── Up 2: 32 channels,  256×256 + skip from Block 1
    │  └── Up 1: 16 channels,  512×512
    │
    ▼ Segmentation Head: Conv2d(16 → 6)
    │
Output: [B, 6, 512, 512] (6 heatmaps, one per landmark)
```

**Key design decisions:**
- **1-channel input:** Cephalograms are grayscale. `smp.Unet` automatically adapts the first conv layer.
- **6-channel output:** One heatmap per landmark (S, N, A, B, Pog, Me).
- **ImageNet pretrained:** Even though ImageNet is RGB natural images, the low-level features (edges, textures) transfer well to medical grayscale images.
- **~24M parameters:** ResNet34 is a good balance of capacity vs. training speed.

**Differential learning rates** via `get_parameter_groups()`:
- Encoder (pretrained): lower LR (1e-4) — fine-tune gently
- Decoder + segmentation head (randomly initialized): higher LR (1e-3) — learn faster

### 5. Training Loop: `trainer.py`

**Purpose:** Encapsulates the complete training loop.

**Device auto-detection:**
```python
def get_device():
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    elif torch.cuda.is_available():         # NVIDIA GPU
        return torch.device("cuda")
    return torch.device("cpu")
```

**Loss function: MSE (Mean Squared Error)**

The model predicts 6 heatmaps; the target is 6 Gaussian heatmaps. MSE computes the average squared difference across all pixels across all 6 channels:

```
Loss = mean((predicted_heatmap - target_heatmap)²)
```

This works because:
- Most of the target is zeros (background)
- The Gaussian peaks provide a strong, localized learning signal
- MSE is smooth and differentiable everywhere

**Optimizer:** Adam with the differential learning rates from `get_parameter_groups()`.

**Scheduler:** `ReduceLROnPlateau(factor=0.5, patience=10)` — if validation loss doesn't improve for 10 consecutive epochs, halve the learning rate. This prevents the model from overshooting and helps it converge to finer precision.

**Checkpointing:** After each epoch, if val loss is the best seen so far, save:
```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_loss": val_loss,
}, "checkpoints/best_model.pth")
```

### 6. MLflow Integration: `mlflow_utils.py`

**Purpose:** Wraps MLflow operations so the training code stays clean.

**What gets logged:**
- **Hyperparameters:** encoder, epochs, batch_size, encoder_lr, decoder_lr, sigma, image_size (via `mlflow.log_params()`)
- **Per-epoch metrics:** train_loss and val_loss with epoch as the step index (creates the loss curves in the MLflow UI)
- **Evaluation metrics:** mre_overall_mm, per-landmark MRE (e.g., `mre_Sella_S_mm`), SDR at all thresholds
- **Artifacts:** The best model checkpoint file

**Metric name sanitization:** MLflow rejects special characters in metric names. The Aariz landmark names contain parentheses like "Sella (S)". The code sanitizes via:
```python
safe_name = re.sub(r"[^a-zA-Z0-9_\-.\s:/]", "", name).strip().replace(" ", "_")
```

**Storage backend:** Uses SQLite (`sqlite:///mlflow.db`) for metrics and metadata storage. This is a single-file database, making it portable and simple.

### 7. Post-Processing: `predict.py`

**Purpose:** Converts raw model output (heatmaps) into precise landmark coordinates.

**Step 1 — Argmax:** Find the pixel with the highest value in each heatmap channel:
```python
flat_idx = heatmap.flatten().argmax()
y, x = divmod(flat_idx, width)
```

**Step 2 — Weighted centroid refinement:** The argmax gives integer pixel coordinates, but the true peak is likely between pixels. The weighted centroid extracts a small window (5×5) around the peak and computes the intensity-weighted center of mass:

```python
def weighted_centroid(heatmap, peak_x, peak_y, window=5):
    patch = heatmap[y1:y2, x1:x2]  # 5×5 window
    total = patch.sum()
    cx = (patch * x_coords).sum() / total  # weighted average x
    cy = (patch * y_coords).sum() / total  # weighted average y
    return cx, cy  # sub-pixel precision
```

This typically improves accuracy by 0.1–0.3mm.

**Step 3 — Confidence scoring:** The peak heatmap value (0–1) directly reflects model confidence:
```python
confidence = min(heatmap.max() * 100, 100.0)  # as percentage
```

A high peak (~0.95) means the model is very certain; a low peak (~0.50) means it's uncertain.

### 8. Angle Computation: `angles.py`

**Purpose:** Computes the three clinical angles from landmark coordinates.

Uses `atan2`-based angle computation for numerical stability:

```python
def compute_angle(point_a, vertex, point_b):
    va = point_a - vertex  # vector from vertex to point A
    vb = point_b - vertex  # vector from vertex to point B
    angle_a = atan2(va[1], va[0])
    angle_b = atan2(vb[1], vb[0])
    angle = abs(angle_a - angle_b)
    if angle > π: angle = 2π - angle  # normalize to [0, π]
    return degrees(angle)
```

**Clinical interpretation** via `interpret_anb()`: Maps ANB values to Class I/II/III classifications with plain-language descriptions:
- ANB < 0°: Class III (underbite)
- ANB 0–1°: Borderline
- ANB 1–4°: Class I (normal)
- ANB 4–7°: Mild Class II (overbite)
- ANB > 7°: Class II (significant overbite)

### 9. Evaluation Pipeline: `evaluation.py`

**Purpose:** Computes clinically meaningful metrics in **millimeters** (not pixels).

**Why pixel spacing matters:** Different X-ray machines have different resolutions. An error of 10 pixels means very different things on different machines:
- Machine A (0.089 mm/px): 10px error = 0.89mm ✅
- Machine B (0.144 mm/px): 10px error = 1.44mm ⚠️

The evaluation pipeline uses per-image pixel spacing from the dataset CSV to convert pixel distances to millimeters before computing MRE:

```python
def compute_mre(pred, gt, pixel_spacings):
    pixel_distances = euclidean_distance(pred, gt)  # in pixels
    mm_distances = pixel_distances * pixel_spacings  # in mm
    return mm_distances.mean()
```

**Coordinate rescaling:** The model operates on 512×512 resized images, but evaluation must happen in the original image resolution. Both predicted and ground truth coordinates are rescaled:
```python
scale_x = original_width / 512
scale_y = original_height / 512
```

**SDR computation:** For each threshold (2.0, 2.5, 3.0, 4.0mm), counts the percentage of all landmark predictions across all test images that fall within that distance of their ground truth.

### 10. Visualization: `overlay.py`

**Purpose:** Creates publication-quality annotated cephalogram figures.

- `draw_landmarks()`: Plots colored circles at predicted positions, "×" markers at GT positions, dashed error lines connecting pred↔GT
- `draw_angle_lines()`: Draws construction lines S–N (yellow), N–A (lime), N–B (orange)
- `create_results_figure()`: Two-panel figure — left: annotated X-ray; right: angle values with normal ranges

Uses `matplotlib.use("Agg")` for headless (server-safe) rendering.

### 11. Streamlit App: `streamlit_app.py`

**Purpose:** Demo-ready web interface for non-specialists.

The app has built-in fallback code — if the `src/` modules can't be imported, it falls back to standalone implementations. This means the app can technically run without the full training pipeline installed.

**Key features:**
- Model checkpoint auto-discovery (scans `checkpoints/` for `.pth` files)
- Sample image loading from `sample_images/`
- Real-time inference with MPS/CUDA/CPU auto-detection
- Color-coded landmark overlay with PIL-based drawing
- Angle metrics with delta-from-normal indicators
- Detailed clinical interpretation covering all 3 angles individually
- Per-landmark confidence score bars with warnings below 70%
- Medical disclaimer

---

## How `run.sh` Works

The `run.sh` script (338 lines) is the single entry point for all project operations. It uses `set -euo pipefail` for strict error handling.

### Configuration (lines 20–30)

```bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"   # absolute path to project root
DOCKER_IMAGE="cephalometric-demo:0.1"          # Docker image name:tag
DOCKER_CONTAINER="cephalometric-demo"          # Docker container name
PORT=8501                                       # Streamlit port
VENV_DIR="${PROJECT_DIR}/.venv"                 # virtual environment path
MLFLOW_PORT=5000                                # MLflow UI port
```

PID files are stored in `/tmp/` (`cephalometric-local.pid`, `cephalometric-mlflow.pid`) to track background processes.

### `./run.sh setup`

1. Creates project directories: `data/`, `checkpoints/`, `mlruns/`, `sample_images/`
2. Creates a Python 3.12 virtual environment at `.venv/` (skips if already exists)
3. Upgrades pip, then installs all dependencies from `requirements.txt`
4. Prints next steps (place data, run training, launch app)

### `./run.sh train [args...]`

1. Verifies `.venv/` exists (errors if not — tells you to run setup first)
2. **Hardware detection:** Runs a quick Python check inside the venv:
   ```python
   import torch; print('yes' if torch.backends.mps.is_available() else 'no')
   ```
   Prints whether MPS (Apple GPU) is available
3. Launches `train.py` with the venv Python, forwarding all CLI arguments:
   ```bash
   "${VENV_DIR}/bin/python" "${PROJECT_DIR}/train.py" "$@"
   ```

### `./run.sh docker build`

1. Verifies Docker is installed
2. Runs `docker build -t cephalometric-demo:0.1 .`
3. The Dockerfile uses `requirements-docker.txt` (CPU-only PyTorch) instead of the full requirements, saving ~2GB and 15 minutes of build time
4. Copies `src/`, `app/`, `.streamlit/`, `checkpoints/`, and `sample_images/` into the image
5. Uses wildcard patterns (`checkpoint[s]/`) so the build doesn't fail if directories are empty

### `./run.sh docker start`

1. Verifies Docker is installed and the image exists
2. Removes any stale container with the same name (`docker rm -f`)
3. Starts a detached container:
   ```bash
   docker run -d --name cephalometric-demo -p 8501:8501 cephalometric-demo:0.1
   ```
4. **Health check loop:** Polls `http://localhost:8501/_stcore/health` every second, up to 20 attempts. Reports success once Streamlit responds.

### `./run.sh docker stop`

Stops and removes the container: `docker stop` → `docker rm`.

### `./run.sh local start`

1. Verifies `.venv/` and `app/streamlit_app.py` exist
2. Checks if Streamlit is already running (via PID file)
3. Launches Streamlit in the background with `nohup`:
   ```bash
   nohup .venv/bin/streamlit run app/streamlit_app.py \
       --server.port=8501 \
       --server.headless=true \
       --browser.gatherUsageStats=false \
       > /tmp/cephalometric-local.log 2>&1 &
   ```
4. Saves the PID to `/tmp/cephalometric-local.pid`
5. **Health check loop:** Same 20-attempt polling pattern as Docker

### `./run.sh local stop`

1. Reads PID from file, kills the process
2. Fallback: if PID file is stale, uses `lsof -ti tcp:8501` to find and kill the process by port

### `./run.sh mlflow start`

1. Creates `mlruns/` directory if needed
2. Verifies `mlflow` binary exists in the venv
3. Launches MLflow UI in the background:
   ```bash
   nohup .venv/bin/mlflow ui \
       --backend-store-uri "sqlite:///${PROJECT_DIR}/mlflow.db" \
       --port 5000 \
       > /tmp/cephalometric-mlflow.log 2>&1 &
   ```
4. The `--backend-store-uri` points to the same SQLite database that the training script writes to, ensuring the UI shows all logged experiments and metrics
5. **Health check loop:** 30 attempts (MLflow can be slow to start due to database initialization)

### `./run.sh mlflow stop`

Same PID/port-based stop pattern as `local stop`, but for port 5000.

### Command Routing (lines 294–337)

The entrypoint uses a `case` statement to dispatch to the correct function:
```bash
case "$MODE" in
    setup)  do_setup ;;
    train)  do_train "$@" ;;
    docker) case "$ACTION" in build|start|stop) ... ;; esac ;;
    local)  case "$ACTION" in start|stop) ... ;; esac ;;
    mlflow) case "$ACTION" in start|stop) ... ;; esac ;;
    *)      usage ;;
esac
```

---

## Setup & Installation

### Prerequisites

- **Python 3.12**
- **Docker** (for containerized deployment)
- macOS (Apple Silicon recommended for GPU training), Linux, or Windows

### Quick Start

```bash
# Clone the repository
git clone https://github.com/thegreatone9/CephalometryLandmarkDetection.git
cd CephalometryLandmarkDetection

# Set up virtual environment + install all dependencies
./run.sh setup

# Place your data in data/ (see Dataset section)

# Train the model (50 epochs, ~50 min on Apple M4)
./run.sh train --epochs 50 --batch-size 4

# Launch the demo app
./run.sh local start
# → http://localhost:8501

# View experiment metrics
./run.sh mlflow start
# → http://localhost:5000
```

---

## Running the App

### Option 1: Docker (Recommended for Demo)

```bash
./run.sh docker build    # Build image (~6 min, CPU-only PyTorch)
./run.sh docker start    # Start container → http://localhost:8501
./run.sh docker stop     # Stop container
```

The Docker image uses CPU-only PyTorch (`requirements-docker.txt`) to keep the image small (~2GB vs ~5GB with CUDA). No GPU required for inference.

### Option 2: Local

```bash
./run.sh local start     # Start with local .venv → http://localhost:8501
./run.sh local stop      # Stop
```

Local mode supports hot-reload — edit code and refresh the browser.

---

## Training a Model

### 1. Prepare the Data

Download the [Aariz dataset](https://www.nature.com/articles/s41597-025-05542-3) and organize:

```
data/
├── cephalogram_machine_mappings.csv
├── train/
│   ├── Cephalograms/           # X-ray images (.png)
│   └── AnnotationsByExperts/
│       ├── junior/             # Junior orthodontist annotations (.json)
│       └── senior/             # Senior orthodontist annotations (.json)
├── val/
│   ├── Cephalograms/
│   └── AnnotationsByExperts/
│       ├── junior/
│       └── senior/
└── test/
    ├── Cephalograms/
    └── AnnotationsByExperts/
        ├── junior/
        └── senior/
```

### 2. Run Training

```bash
# Smoke test (verify pipeline works)
./run.sh train --epochs 2 --batch-size 4

# Full training
./run.sh train --epochs 50 --batch-size 4

# Custom configuration
./run.sh train --epochs 100 --batch-size 8 --encoder resnet34 --img-size 512
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 4 | Batch size (4 is safe for 16GB unified memory) |
| `--img-size` | 512 | Input image resolution |
| `--encoder` | resnet34 | Encoder backbone |
| `--encoder-lr` | 1e-4 | Encoder learning rate (lower for pretrained weights) |
| `--decoder-lr` | 1e-3 | Decoder learning rate (higher for random init) |
| `--sigma` | 5.0 | Gaussian heatmap sigma (pixel spread of targets) |
| `--data-dir` | data | Path to dataset |
| `--checkpoint-dir` | checkpoints | Path to save model weights |

### Hardware Acceleration

| Platform | Device | Expected Speed |
|----------|--------|---------------|
| Apple Silicon (M1/M2/M3/M4) | MPS | ~60s/epoch |
| NVIDIA GPU | CUDA | ~20–40s/epoch |
| CPU only | CPU | ~5–10min/epoch |

---

## Experiment Tracking with MLflow

```bash
./run.sh mlflow start    # → http://localhost:5000
./run.sh mlflow stop     # Stop the UI
```

### What Gets Logged

| Category | Metrics |
|----------|---------|
| **Hyperparameters** | encoder, epochs, batch_size, learning rates, sigma, image size |
| **Per-epoch** | train_loss, val_loss (as step-indexed curves) |
| **Evaluation** | mre_overall_mm, per-landmark MRE, SDR at 2/2.5/3/4mm |
| **Artifacts** | Best model checkpoint (best_model.pth) |
| **Run naming** | `{encoder}-ep{epochs}-bs{batch}-img{size}` |

MLflow stores all data in a local SQLite database (`mlflow.db`). The `run.sh mlflow start` command connects the MLflow UI to this same database via `--backend-store-uri sqlite:///mlflow.db`.

---

## Dataset

This project uses the **Aariz dataset**, published in Nature Scientific Data:

> Tahmasebi, A., et al. "Aariz: A Benchmark Dataset for Automatic Cephalometric Landmark Detection and CVM Stage Classification." *Scientific Data* 12, 579 (2025). [https://doi.org/10.1038/s41597-025-05542-3](https://doi.org/10.1038/s41597-025-05542-3)

| Property | Value |
|----------|-------|
| **Total images** | 1,000 lateral cephalograms |
| **Split** | 700 train / 150 val / 150 test |
| **Landmarks annotated** | 29 (we use 6) |
| **Annotators** | Junior + Senior orthodontists |
| **Image format** | PNG, various resolutions |
| **Pixel spacing** | 5 unique values (0.089–0.144 mm/px) |
| **Annotation format** | JSON with symbol-based lookup |

---

## Acknowledgements

- **Dataset:** [Aariz — A Benchmark Dataset for Automatic Cephalometric Landmark Detection](https://www.nature.com/articles/s41597-025-05542-3) (Nature Scientific Data, 2025)
- **Model backbone:** [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) by Pavel Iakubovskii
- **Experiment tracking:** [MLflow](https://mlflow.org/)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **GPU acceleration:** Apple Metal Performance Shaders (MPS)
