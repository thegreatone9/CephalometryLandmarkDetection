# 🦷 AI-Powered Cephalometric Landmark Detection

**Automated orthodontic analysis using deep learning to detect anatomical landmarks on lateral skull X-rays and compute clinically meaningful jaw alignment measurements.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## Table of Contents

- [Tools & Technologies Used](#tools--technologies-used)
- [Cephalometric Analysis: The Domain](#cephalometric-analysis-the-domain)
  - [What Is a Cephalogram?](#what-is-a-cephalogram)
  - [What Are Landmarks?](#what-are-landmarks)
  - [What Are the Angles and Why Do They Matter?](#what-are-the-angles-and-why-do-they-matter)
  - [The Manual Process and Why Automation Helps](#the-manual-process-and-why-automation-helps)
- [What This Project Does](#what-this-project-does)
- [Results & Performance](#results--performance)
- [Project Structure](#project-structure)
- [Code Architecture Deep-Dive](#code-architecture-deep-dive)
  - [The Training Data Flow](#the-training-data-flow)
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

## Tools & Technologies Used

The following table summarizes each tool in the stack and its role in the project:

| Tool | What It Is | Why We Use It |
|------|-----------|---------------|
| **Python 3.12** | A general-purpose programming language | The primary language for all code in this project |
| **PyTorch** | An open-source machine learning framework (like a toolbox for building AI models) | Provides the building blocks for our neural network — tensors, layers, optimizers, GPU support |
| **segmentation-models-pytorch** | A library of pre-built neural network architectures designed for image analysis | Gives us the U-Net model architecture out of the box, so we don't have to build it from scratch |
| **Streamlit** | A Python framework for building web apps with minimal code | Powers our demo web interface — lets users upload X-rays and see results without any frontend coding |
| **MLflow** | An experiment tracking tool for machine learning | Records every training run — what settings we used, how the model performed, which model file was saved — so we can compare experiments |
| **Docker** | A tool that packages an application and all its dependencies into a portable "container" | Lets anyone run the demo app on any computer without installing Python, PyTorch, etc. — just `docker run` and it works |
| **Albumentations** | An image augmentation library | Applies controlled random modifications to training images (rotation, brightness) to make the model more robust |
| **NumPy** | A Python library for numerical computing with arrays | Used for matrix math, coordinate calculations, and image manipulation |
| **Matplotlib** | A Python plotting and visualization library | Creates annotated X-ray figures with landmark overlays |

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

**What it measures:** The angle at Nasion between lines drawn to Sella and A-point. This indicates how far forward or backward the **upper jaw** sits relative to the skull base.

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

> **MRE** (Mean Radial Error) = average Euclidean distance between predicted and actual landmark positions, converted to millimeters using per-image pixel spacing metadata from the X-ray machine.
>
> **SDR** (Successful Detection Rate) = percentage of landmarks that fall within a given distance of the correct position.
>
> The **clinical acceptability threshold is 2mm** — our model places **94.9% of landmarks within this tolerance**.

### Per-Landmark Accuracy

| Landmark | MRE (mm) | Notes |
|----------|----------|-------|
| Menton (Me) | 0.51 | Easiest — sharp bone edge, high contrast |
| Pogonion (Pog) | 0.61 | Clear convexity at chin prominence |
| Sella (S) | 0.73 | Well-defined cavity in the skull base |
| B-point (B) | 0.92 | Subtle concavity, harder to distinguish |
| Nasion (N) | 0.94 | Multiple overlapping structures in that area |
| A-point (A) | 0.98 | Subtle concavity, hardest landmark |

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
│   │   ├── dataset.py             # Loads images + annotations, generates heatmap targets
│   │   └── preprocessing.py       # Image augmentation pipelines
│   ├── models/
│   │   └── unet.py                # Neural network model definition
│   ├── training/
│   │   ├── trainer.py             # Training loop: feeds data, computes loss, updates model
│   │   └── mlflow_utils.py        # Experiment tracking helpers
│   ├── inference/
│   │   ├── predict.py             # Converts model output to coordinates + confidence
│   │   └── angles.py              # Angle computation + clinical interpretation
│   ├── viz/
│   │   └── overlay.py             # Draws landmarks on X-ray images
│   └── evaluation.py              # Computes accuracy metrics (MRE, SDR)
├── train.py                       # Main script to run training
├── run.sh                         # Unified operations script (start/stop/build everything)
├── Dockerfile                     # Defines how to package the app as a Docker container
├── .dockerignore                  # Tells Docker which files to skip when building
├── requirements.txt               # Python packages needed (with GPU support for training)
├── requirements-docker.txt        # Python packages needed (CPU-only, for Docker)
├── .streamlit/                    # Streamlit visual theme settings
├── checkpoints/                   # Saved model files (best_model.pth, ~280MB)
├── sample_images/                 # Demo X-rays for the web app
└── data/                          # Training data (not committed to git — too large)
```

### How the Modules Connect

```
train.py (orchestrator — runs the full pipeline)
├── src/data/dataset.py ──── Loads images + annotations, generates training targets
│   └── src/data/preprocessing.py ──── Augments images (rotation, brightness, etc.)
├── src/models/unet.py ──── Defines the neural network architecture
├── src/training/trainer.py ──── The training loop (forward pass → loss → backprop → save)
│   ├── src/models/unet.py
│   └── src/training/mlflow_utils.py ──── Records metrics to MLflow
├── src/training/mlflow_utils.py ──── Sets up experiment, logs hyperparameters
└── src/evaluation.py ──── Tests the trained model, computes MRE and SDR
    └── src/inference/predict.py ──── Converts heatmaps to coordinates

app/streamlit_app.py (standalone web app for inference)
├── src/models/unet.py ──── Loads the trained model
├── src/inference/predict.py ──── Runs inference on uploaded images
└── src/inference/angles.py ──── Computes and interprets angles
```

---

## Code Architecture Deep-Dive

### The Training Data Flow

The complete journey from raw data to a trained model follows this pipeline:

```
Raw X-ray PNG + JSON annotations
    ↓ dataset.py: Parse JSON, average junior + senior expert annotations
    ↓ dataset.py: Generate 6-channel Gaussian heatmaps (the "target" the model learns to produce)
    ↓ preprocessing.py: Apply random augmentation (rotation, brightness)
    ↓ PyTorch DataLoader: Group into batches of 4 images
    ↓ unet.py: Feed batch through U-Net → predicted heatmaps [4, 6, 512, 512]
    ↓ trainer.py: Compute MSE loss (how far off are the predictions?)
    ↓ trainer.py: Backpropagation (figure out how to improve)
    ↓ trainer.py: Optimizer step (nudge all parameters to reduce loss)
    ↓ trainer.py: Repeat for all batches = 1 epoch
    ↓ trainer.py: Save checkpoint if this epoch's validation loss is the best so far
    ↓ (repeat for 50 epochs)
    ↓ predict.py: Find peaks in heatmaps → (x, y) coordinates
    ↓ evaluation.py: Convert pixel distances to millimeters → compute MRE and SDR
    ↓ mlflow_utils.py: Log everything for posterity
    ↓ Best model saved to checkpoints/best_model.pth
```

### 1. Training Entry Point: `train.py`

**Purpose:** The top-level script that orchestrates everything. Running `./run.sh train --epochs 50` invokes this module.

**What happens step by step:**

1. **Parse arguments** via `parse_args()` — reads command-line settings like encoder architecture, number of epochs, batch size, learning rates, heatmap sigma, image size, and data/checkpoint directory paths.

2. **Load pixel spacings** from `data/cephalogram_machine_mappings.csv`. This CSV maps each X-ray image to the physical pixel size of the machine that captured it (in mm/pixel). Different X-ray machines have different resolutions — values range from 0.089 to 0.144 mm/px across 5 different machines in the dataset. This is critical for computing errors in real-world millimeters rather than arbitrary pixels. If the CSV isn't found, all images default to 0.1 mm/px.

3. **Build DataLoaders** via `build_dataloaders()` — creates three `CephalometricDataset` instances for `train/`, `val/`, and `test/` splits. A **DataLoader** is a PyTorch utility that wraps a dataset and handles batching (grouping images into groups of 4), shuffling (randomizing order each epoch for training), and feeding data to the model efficiently. The training loader shuffles; validation/test loaders do not.

4. **Set up MLflow** — creates or retrieves an experiment called `cephalometric-landmark-detection`, starts a tracked run named like `resnet34-ep50-bs4-img512`, and logs all hyperparameters (settings).

5. **Train** — creates a `Trainer` object and calls `trainer.fit(train_loader, val_loader)`. This runs the full training loop across all epochs and returns the path to the best-performing checkpoint.

6. **Evaluate** — loads the best checkpoint, runs the model on the test set, computes MRE/SDR metrics, logs them to MLflow, and prints a formatted results table.

### 2. Dataset & Heatmap Generation: `dataset.py`

**Purpose:** The most complex module — handles loading the Aariz dataset's unique format, averaging annotations from two expert groups, and generating the heatmap targets that the model learns to predict.

**Constants defined here:**
- `ALL_LANDMARK_SYMBOLS`: All 29 landmarks in the Aariz dataset
- `SELECTED_SYMBOLS = ["S", "N", "A", "B", "Pog", "Me"]`: The 6 we use
- `NUM_LANDMARKS = 6`

**JSON annotation parsing:** The Aariz dataset stores landmark positions in JSON files with a nested structure:

```json
{
  "landmarks": [
    {"symbol": "S", "value": {"x": 234.5, "y": 189.2}},
    {"symbol": "N", "value": {"x": 312.1, "y": 201.8}},
    ...
  ]
}
```

The function `_load_landmarks_from_json()` reads this file and converts it into a simple Python dictionary mapping each landmark symbol to its (x, y) position.

**Dual-annotator averaging:** Each image has annotations from both a junior and senior orthodontist, stored in separate JSON files. The function `_average_annotations()` loads both, and for each landmark, computes the midpoint of the two experts' positions. This produces more reliable **ground truth** (the "correct answer" the model trains against) than using either expert alone.

**Why heatmaps instead of direct coordinate prediction:**

A naive approach would be to have the model directly output 12 numbers (x and y for each of 6 landmarks). But this has a fundamental problem: a small error in any one number (say, off by 20 pixels) produces the same loss regardless of whether the prediction is close or far from the correct answer in 2D space. The loss landscape (the "terrain" the optimizer navigates) is flat and hard to learn from.

Instead, we use **heatmap regression**. For each landmark, we create a 512×512 "heatmap" image where:
- The correct landmark position has a bright peak (value = 1.0)
- The brightness fades smoothly outward in a bell-curve shape (a **Gaussian** distribution)
- Everything far from the landmark is dark (value ≈ 0)

```
_generate_gaussian_heatmap(height, width, cx, cy, sigma=5.0):
    # Creates a 2D bell-curve with peak=1.0 at (cx, cy)
    # sigma (σ) controls how wide the bell curve is
    # σ=5.0 means the bright spot spans roughly 10 pixels across
    # Values fade to near-zero beyond ~15 pixels from center
    heatmap[y, x] = exp(-((x - cx)² + (y - cy)²) / (2σ²))
```

The Gaussian shape provides a smooth, gradual signal for the loss function: even if a prediction is off by a few pixels, the overlap with the target peak is nonzero, which gives the optimizer a useful gradient to work with.

The function `generate_heatmaps()` creates 6 of these heatmaps (one per landmark), stacked into a single tensor of shape `[6, 512, 512]`.

**What `__getitem__()` returns** (called every time the DataLoader needs one sample):
- `image`: the X-ray as a tensor `[1, 512, 512]` — 1 channel (grayscale), normalized to values between 0 and 1
- `heatmaps`: the target heatmaps as a tensor `[6, 512, 512]` — what the model should produce
- `landmarks`: the raw (x, y) coordinates as a tensor `[6, 2]` — used later for evaluation
- `meta`: a dictionary with metadata — image path, cephalogram ID, original image size, resized size, and pixel spacing (mm/px)

### 3. Data Augmentation: `preprocessing.py`

**Purpose:** Defines the image augmentation pipelines using **Albumentations**, a popular image augmentation library.

**The key challenge:** When an image is rotated or shifted, the landmark coordinates must be transformed in exactly the same way, or the targets become wrong. Albumentations solves this with **keypoint-aware transforms** — we register the (x, y) points as attached to the image, and Albumentations automatically adjusts them when the image is modified.

**Training augmentations (applied randomly each time an image is loaded):**
- `Resize(512, 512)` — standardize all images to the same size
- `Rotate(limit=±15°, p=0.5)` — 50% chance of rotating up to 15 degrees (simulates head tilt)
- `RandomBrightnessContrast(±0.15, p=0.5)` — 50% chance of brightness/contrast change (simulates different X-ray exposures)
- `GaussNoise(std_range=(0.01, 0.05), p=0.3)` — 30% chance of adding slight random noise (simulates sensor imperfections)

**Validation/test augmentations:** Only `Resize(512, 512)` — no random modifications, because we want consistent evaluation.

**Critically omitted: horizontal flip.** Flipping is normally a "free" augmentation for natural images (a flipped cat is still a cat). But cephalograms are **always** taken from the same side — flipping one would create an anatomically impossible mirror image and confuse the model about left vs. right anatomy.

### 4. Model Architecture: `unet.py`

**Purpose:** Defines the neural network architecture that processes X-ray images and outputs heatmaps.

**What is a U-Net?**

U-Net is a neural network architecture originally designed for medical image analysis (specifically, segmenting cell boundaries under microscopes). Its name comes from its U-shaped structure:

1. **Encoder (left side of the U):** Progressively shrinks the image while extracting increasingly abstract features — from edges to shapes to anatomical structures — while losing spatial detail.

2. **Decoder (right side of the U):** Progressively enlarges back to the original resolution, using the abstract understanding to make precise, pixel-level predictions.

3. **Skip connections (the bridges across the U):** Direct wires connecting each encoder level to the corresponding decoder level. These let the decoder access the fine spatial details that the encoder captured before shrinking. Without skip connections, the decoder would have to reconstruct precise positions from very small, abstract representations — skip connections dramatically improve spatial precision.

**What is ResNet34?**

ResNet34 is a specific encoder architecture with 34 layers, designed by Microsoft Research. We don't build it from scratch — we use it as a pre-built component from the `segmentation_models_pytorch` library. The "34" refers to the number of layers deep it goes. It's been pretrained on **ImageNet** (14 million everyday photographs), meaning it already knows how to detect edges, textures, and shapes before we even show it a single X-ray.

**Full architecture diagram:**

```
Input: [B, 1, 512, 512]
       B images, 1 channel (grayscale), 512×512 pixels
    │
    ▼ ResNet34 Encoder (pretrained — already knows edges, textures, shapes)
    │  ├── Block 1: 64 feature maps,  256×256 pixels (detects edges)
    │  ├── Block 2: 64 feature maps,  128×128 pixels (detects textures)
    │  ├── Block 3: 128 feature maps, 64×64 pixels  (detects shapes)
    │  ├── Block 4: 256 feature maps, 32×32 pixels  (detects structures)
    │  └── Block 5: 512 feature maps, 16×16 pixels  (detects anatomy — "bottleneck")
    │         ↕ skip connections carry spatial detail across
    ▼ U-Net Decoder (randomly initialized — learns from scratch)
    │  ├── Up 5: 256 maps, 32×32 + skip from Block 4
    │  ├── Up 4: 128 maps, 64×64 + skip from Block 3
    │  ├── Up 3: 64 maps,  128×128 + skip from Block 2
    │  ├── Up 2: 32 maps,  256×256 + skip from Block 1
    │  └── Up 1: 16 maps,  512×512
    │
    ▼ Segmentation Head: a single convolutional layer (16 → 6)
    │  (Convolution = sliding a small filter across the image to transform features)
    │
Output: [B, 6, 512, 512]
        B images, 6 heatmaps (one per landmark), 512×512 pixels
```

**A "feature map"** (also called a "channel") is like a filtered version of the image that highlights a specific pattern. Block 1 might have 64 feature maps detecting 64 types of edges; Block 5 might have 512 feature maps each detecting a different anatomical structure.

**Why ~24 million parameters?** Each "parameter" is a single number inside the model that gets tuned during training. ResNet34 is a good balance — small enough to train in ~50 minutes on a laptop GPU, large enough to learn the complex patterns in cephalometric anatomy.

**Differential learning rates:** Since the encoder is pretrained (already knows useful patterns), we update it slowly (learning rate 1e-4 = 0.0001) to preserve what it already knows. The decoder starts from random values and needs to learn everything from scratch, so we update it faster (learning rate 1e-3 = 0.001). The function `get_parameter_groups()` splits the model's parameters into these two groups for the optimizer.

### 5. Training Loop: `trainer.py`

**Purpose:** Contains the `Trainer` class that runs the actual training process — the repeated cycle of predict → measure error → improve.

**Device auto-detection:** Before training starts, the code checks what hardware is available:
```python
def get_device():
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    elif torch.cuda.is_available():         # NVIDIA GPU
        return torch.device("cuda")
    return torch.device("cpu")              # Fallback — much slower
```

**The loss function: MSE (Mean Squared Error)**

After the model produces predicted heatmaps, we need a single number that represents "how wrong" the predictions are. That's the **loss function**. We use MSE, which computes the average of all squared pixel differences between the predicted and target heatmaps:

```
Loss = mean((predicted_heatmap - target_heatmap)²)
```

Squaring the differences means large errors are penalized much more than small ones (an error of 10 produces a loss of 100, but an error of 1 produces a loss of only 1). This pushes the model to eliminate big mistakes first.

**The optimizer: Adam**

After computing the loss, **backpropagation** calculates how each of the 24 million parameters contributed to the error. Then the **optimizer** adjusts each parameter to reduce the loss. Adam is a popular optimizer that keeps a running average of past gradients to adapt its step size — parameters that consistently point in the same direction get larger steps, while noisy parameters get smaller steps.

**The scheduler: ReduceLROnPlateau**

Sometimes training gets stuck — the loss stops decreasing. The **learning rate scheduler** monitors the validation loss, and if it hasn't improved for 10 consecutive epochs (called "patience"), it halves the learning rate. A smaller learning rate means smaller, more careful steps, which can help the model escape a stuck point and find finer precision.

**Checkpointing:** After every epoch, the trainer compares the current validation loss to the best seen so far. If it's better, it saves the model's complete state to `checkpoints/best_model.pth`:
```python
torch.save({
    "epoch": epoch,                          # which epoch this was
    "model_state_dict": model.state_dict(),  # all 24M learned parameters
    "optimizer_state_dict": optimizer.state_dict(),
    "val_loss": val_loss,                    # how good this model is
}, "checkpoints/best_model.pth")
```

This ensures we always keep the best-performing version of the model, even if later epochs overfit and get worse.

### 6. MLflow Integration: `mlflow_utils.py`

**Purpose:** Handles experiment tracking — recording what settings were used and how the model performed, so different training runs can be compared.

MLflow is an open-source experiment tracking tool that acts as a structured lab notebook for machine learning. Each training run records:
- **Hyperparameters:** The settings used (encoder type, learning rate, number of epochs, etc.)
- **Metrics:** Model performance at each epoch (training loss, validation loss) and final evaluation results (MRE, SDR)
- **Artifacts:** The trained model file

The MLflow web UI (at http://localhost:5000) allows side-by-side comparison of runs, loss curve visualization, and identification of the best-performing configuration.

**Key functions:**
- `setup_experiment()`: Creates or finds the MLflow experiment
- `log_hyperparams()`: Records the training settings
- `log_epoch_metrics()`: Records train/val loss after each epoch (these become the loss curves in the UI)
- `log_evaluation_metrics()`: Records final MRE and SDR numbers
- `log_model_artifact()`: Uploads the `.pth` model file to MLflow's artifact store

**Metric name sanitization:** MLflow doesn't allow special characters like parentheses in metric names. The Aariz landmark names include them (e.g., "Sella (S)"), so the code strips them out via regex before logging.

**Storage:** Everything is stored in a local SQLite database file (`mlflow.db`), which is just a single file on disk. No external database server needed.

### 7. Post-Processing: `predict.py`

**Purpose:** The model outputs 6 heatmaps (512×512 images of "brightness"). We need to convert these into precise (x, y) coordinates for each landmark. This module does that conversion.

**Step 1 — Argmax (find the brightest pixel):**

For each of the 6 heatmaps, find the pixel with the highest value. That pixel's position is our initial landmark coordinate:

```python
flat_idx = heatmap.flatten().argmax()  # index of brightest pixel
y, x = divmod(flat_idx, width)         # convert flat index to 2D coordinates
```

**"Argmax"** literally means "argument of the maximum" — it returns the *position* of the maximum value, not the value itself.

**Step 2 — Weighted centroid refinement (sub-pixel precision):**

The argmax gives us integer pixel coordinates (e.g., x=234, y=189), but the true peak might be between pixels (e.g., x=234.3, y=189.7). To get this **sub-pixel precision**, we look at a small 5×5 patch of pixels around the peak and compute a brightness-weighted average position:

```python
def weighted_centroid(heatmap, peak_x, peak_y, window=5):
    patch = heatmap[y1:y2, x1:x2]  # extract 5×5 window around peak
    total = patch.sum()              # total brightness
    cx = sum(x * brightness) / total # brightness-weighted average x
    cy = sum(y * brightness) / total # brightness-weighted average y
    return cx, cy                    # e.g., (234.3, 189.7)
```

This is analogous to computing the center of gravity of the brightness distribution. It typically improves accuracy by 0.1–0.3mm.

**Step 3 — Confidence scoring:**

The peak brightness value (between 0 and 1) directly reflects how confident the model is about that landmark's position:
```python
confidence = min(heatmap.max() * 100, 100.0)  # convert to percentage
```

A tall, sharp peak (~95%) means the model is very certain about the position. A low, spread-out peak (~50%) means it's uncertain — maybe the anatomy is ambiguous in that area.

### 8. Angle Computation: `angles.py`

**Purpose:** Once we have the (x, y) positions of the 6 landmarks, this module computes the three clinical angles (SNA, SNB, ANB) using basic trigonometry.

The angle between two lines meeting at a vertex is computed using `atan2` (a standard trigonometric function that handles all quadrants correctly):

```python
def compute_angle(point_a, vertex, point_b):
    va = point_a - vertex  # vector from vertex to point A
    vb = point_b - vertex  # vector from vertex to point B
    angle_a = atan2(va[1], va[0])  # direction of vector A
    angle_b = atan2(vb[1], vb[0])  # direction of vector B
    angle = abs(angle_a - angle_b)
    if angle > π: angle = 2π - angle  # ensure angle is in [0°, 180°]
    return degrees(angle)
```

- `compute_sna(sella, nasion, a_point)` → angle at Nasion between S and A
- `compute_snb(sella, nasion, b_point)` → angle at Nasion between S and B
- `compute_anb(sna, snb)` → simply SNA minus SNB

The `interpret_anb()` function maps the ANB value to a clinical classification (Class I/II/III) with a plain-language description.

### 9. Evaluation Pipeline: `evaluation.py`

**Purpose:** Computes how accurate the trained model is, using clinically meaningful metrics measured in **millimeters** (not pixels).

**Why pixel spacing matters:** Different X-ray machines have different physical pixel sizes. An error of 10 pixels means very different things on different machines:
- Machine A (0.089 mm/px): 10px error = 0.89mm ✅ (within clinical tolerance)
- Machine B (0.144 mm/px): 10px error = 1.44mm ⚠️ (borderline)

The evaluation pipeline looks up each image's pixel spacing from the dataset metadata and multiplies:

```python
def compute_mre(pred, gt, pixel_spacings):
    pixel_distance = sqrt((pred_x - gt_x)² + (pred_y - gt_y)²)  # in pixels
    mm_distance = pixel_distance * pixel_spacing                   # in millimeters
    return mean(mm_distance)  # average across all landmarks and images
```

**Coordinate rescaling:** The model operates on 512×512 resized images, but the original images may be different sizes (e.g., 2048×2048). Before computing errors, both predicted and ground truth coordinates are rescaled back to the original resolution:
```python
scale_x = original_width / 512
scale_y = original_height / 512
```

**SDR computation:** For each threshold (2.0mm, 2.5mm, 3.0mm, 4.0mm), counts the percentage of all landmark predictions (across all test images) that fall within that distance of their ground truth position.

### 10. Visualization: `overlay.py`

**Purpose:** Creates publication-quality annotated X-ray images with landmarks and angle lines drawn on them.

- `draw_landmarks()`: Plots colored circles at predicted positions, "×" markers at ground truth positions, and dashed error lines connecting them
- `draw_angle_lines()`: Draws the three construction lines (S–N, N–A, N–B) used for angle measurement
- `create_results_figure()`: Creates a two-panel figure — left panel has the annotated X-ray; right panel shows computed angles with normal ranges

Uses `matplotlib.use("Agg")` for **headless rendering** — generating images without needing a screen (important for running on servers or inside Docker containers).

### 11. Streamlit App: `streamlit_app.py`

**Purpose:** The user-facing web interface that lets anyone upload an X-ray and see results without any coding or ML knowledge.

**Key features:**
- **Model auto-discovery:** Scans the `checkpoints/` folder for `.pth` model files
- **Sample image loading:** Provides pre-loaded X-rays from `sample_images/`
- **Real-time inference:** Runs the model on the uploaded image (using MPS/CUDA/CPU)
- **Visualization:** Draws color-coded landmarks and angle lines on the X-ray using PIL (Python Imaging Library)
- **Angle metrics:** Shows SNA, SNB, ANB with delta-from-normal indicators
- **Clinical interpretation:** Detailed, plain-English explanation of all three angles — covering what each means, whether it's normal, and what it implies
- **Confidence scores:** Per-landmark confidence bars with ⚠️ warnings below 70%
- **Medical disclaimer:** Reminds users this is for demonstration only

**Fallback design:** The app includes standalone implementations of key functions (preprocessing, angle computation, drawing). If the `src/` modules can't be imported, the app still works using these fallbacks. This means the Docker container doesn't need the full training pipeline installed.

---

## How `run.sh` Works

The `run.sh` script (338 lines of Bash) is the single entry point for all project operations. Here's exactly what each command does internally.

### Script Configuration (lines 20–30)

```bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"   # absolute path to project root
DOCKER_IMAGE="cephalometric-demo:0.1"          # Docker image name:tag
DOCKER_CONTAINER="cephalometric-demo"          # Docker container name
PORT=8501                                       # Streamlit port
VENV_DIR="${PROJECT_DIR}/.venv"                 # Python virtual environment path
MLFLOW_PORT=5000                                # MLflow UI port
```

Process IDs (PIDs) for background processes are saved to `/tmp/` so the script can stop them later.

### `./run.sh setup` — Environment Setup

A **virtual environment** (venv) is an isolated Python installation that keeps this project's packages separate from the system Python. This prevents version conflicts with other projects.

1. Creates project directories: `data/`, `checkpoints/`, `mlruns/`, `sample_images/`
2. Creates a Python 3.12 virtual environment at `.venv/` (skips if already exists)
3. Upgrades pip (Python's package installer), then installs all packages listed in `requirements.txt`
4. Prints next steps

### `./run.sh train [args...]` — Model Training

1. Verifies `.venv/` exists (errors out with a helpful message if not)
2. **Hardware detection:** Runs a quick Python check:
   ```python
   import torch; print('yes' if torch.backends.mps.is_available() else 'no')
   ```
   Prints whether Apple GPU acceleration is available
3. Launches the training script, forwarding all command-line arguments:
   ```bash
   "${VENV_DIR}/bin/python" "${PROJECT_DIR}/train.py" "$@"
   ```

### `./run.sh docker build` — Build Docker Image

A Docker image is a snapshot of a complete computer environment — OS, Python, all packages, and application code — packaged into a single file that can run anywhere.

1. Verifies Docker is installed
2. Runs `docker build -t cephalometric-demo:0.1 .` which:
   - Starts from a minimal Linux image with Python 3.12
   - Installs system libraries for image processing
   - Installs Python packages from `requirements-docker.txt` (**CPU-only PyTorch** — saves ~2GB vs the full GPU version, since Docker is only used for inference)
   - Copies the `src/`, `app/`, `.streamlit/`, `checkpoints/`, and `sample_images/` folders into the image
   - Uses wildcard patterns (`checkpoint[s]/`) so the build doesn't fail if directories are empty
3. Reports success

### `./run.sh docker start` — Run Docker Container

A container is a running instance of an image — like starting a virtual computer from the snapshot.

1. Verifies Docker is installed and the image exists
2. Removes any stale container with the same name
3. Starts a detached container (runs in the background), mapping port 8501:
   ```bash
   docker run -d --name cephalometric-demo -p 8501:8501 cephalometric-demo:0.1
   ```
4. **Health check loop:** Polls `http://localhost:8501/_stcore/health` every second (up to 20 attempts) until Streamlit responds, then prints the URL

### `./run.sh docker stop` — Stop Docker Container

Stops and removes the running container: `docker stop` → `docker rm`.

### `./run.sh local start` — Run Streamlit Locally

1. Verifies the virtual environment and app file exist
2. Checks if Streamlit is already running (via saved PID file)
3. Launches Streamlit in the background:
   ```bash
   nohup .venv/bin/streamlit run app/streamlit_app.py \
       --server.port=8501 \
       --server.headless=true \
       --browser.gatherUsageStats=false \
       > /tmp/cephalometric-local.log 2>&1 &
   ```
   - `nohup` prevents the process from dying when the terminal is closed
   - `--server.headless=true` suppresses the "open browser" prompt
   - Output is logged to `/tmp/cephalometric-local.log`
4. Saves the process ID to `/tmp/cephalometric-local.pid`
5. Same health check polling loop as Docker

### `./run.sh local stop` — Stop Local Streamlit

1. Reads the PID from the saved file and kills the process
2. **Fallback:** If the PID file is stale (process already died), uses `lsof -ti tcp:8501` to find any process listening on port 8501 and kills it

### `./run.sh mlflow start` — Launch MLflow UI

1. Creates the `mlruns/` directory if needed
2. Verifies the `mlflow` command exists in the virtual environment
3. Launches the MLflow UI server in the background:
   ```bash
   nohup .venv/bin/mlflow ui \
       --backend-store-uri "sqlite:///${PROJECT_DIR}/mlflow.db" \
       --port 5000 \
       > /tmp/cephalometric-mlflow.log 2>&1 &
   ```
   The `--backend-store-uri` flag points the UI to the same SQLite database that the training script writes metrics to. This is what connects the UI to the training runs.
4. Health check loop (30 attempts — MLflow can be slower to start than Streamlit)

### `./run.sh mlflow stop` — Stop MLflow UI

Same PID/port-based stop pattern as `local stop`, but for port 5000.

### Command Routing (lines 294–337)

The script's entrypoint uses a `case` statement to dispatch the given command to the correct function:
```bash
case "$MODE" in
    setup)  do_setup ;;
    train)  do_train "$@" ;;
    docker) case "$ACTION" in build|start|stop) ... ;; esac ;;
    local)  case "$ACTION" in start|stop) ... ;; esac ;;
    mlflow) case "$ACTION" in start|stop) ... ;; esac ;;
    *)      usage ;;  # print help
esac
```

---

## Setup & Installation

### Prerequisites

- **Python 3.12**
- **Docker** (only needed for containerized deployment)
- macOS (Apple Silicon recommended for GPU training), Linux, or Windows

### Quick Start

```bash
# Clone the repository
git clone https://github.com/thegreatone9/CephalometryLandmarkDetection.git
cd CephalometryLandmarkDetection

# Set up virtual environment + install all dependencies
./run.sh setup

# Place data in data/ (see Dataset section)

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

The Docker image uses CPU-only PyTorch (`requirements-docker.txt`) to keep the image small (~2GB vs ~5GB with GPU libraries). GPU is not needed for inference — it's fast enough on CPU.

### Option 2: Local

```bash
./run.sh local start     # Start with local .venv → http://localhost:8501
./run.sh local stop      # Stop
```

Local mode supports **hot-reload** — edit code and refresh the browser to see changes immediately.

---

## Training a Model

### 1. Prepare the Data

Download the [Aariz dataset](https://www.nature.com/articles/s41597-025-05542-3) and organize:

```
data/
├── cephalogram_machine_mappings.csv    # Pixel spacing per X-ray machine
├── train/
│   ├── Cephalograms/                   # X-ray images (.png)
│   └── AnnotationsByExperts/
│       ├── junior/                     # Junior orthodontist annotations (.json)
│       └── senior/                     # Senior orthodontist annotations (.json)
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
# Smoke test (verify the pipeline works end-to-end)
./run.sh train --epochs 2 --batch-size 4

# Full training
./run.sh train --epochs 50 --batch-size 4

# Custom configuration
./run.sh train --epochs 100 --batch-size 8 --encoder resnet34 --img-size 512
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | How many times to loop through all training images |
| `--batch-size` | 4 | How many images to process at once (4 is safe for 16GB memory) |
| `--img-size` | 512 | Resolution to resize all images to (pixels) |
| `--encoder` | resnet34 | Which pretrained encoder backbone to use |
| `--encoder-lr` | 1e-4 | Learning rate for pretrained encoder (small — preserve existing knowledge) |
| `--decoder-lr` | 1e-3 | Learning rate for decoder (larger — learn from scratch) |
| `--sigma` | 5.0 | How wide the Gaussian heatmap peaks are (in pixels) |
| `--data-dir` | data | Path to dataset |
| `--checkpoint-dir` | checkpoints | Path to save model files |

### Hardware Acceleration

| Platform | GPU Backend | Expected Speed |
|----------|-------------|---------------|
| Apple Silicon (M1/M2/M3/M4) | MPS (Metal) | ~60s/epoch |
| NVIDIA GPU | CUDA | ~20–40s/epoch |
| CPU only | — | ~5–10min/epoch |

---

## Experiment Tracking with MLflow

```bash
./run.sh mlflow start    # → http://localhost:5000
./run.sh mlflow stop     # Stop the UI
```

### What You'll See in the MLflow UI

- **Run list:** Each training run appears as a row with its name (e.g., `resnet34-ep50-bs4-img512`)
- **Parameters:** All hyperparameters used for that run
- **Metrics:** Click a run to see loss curves (train_loss and val_loss over epochs), final MRE, and SDR values
- **Artifacts:** Download the trained model file directly from the UI

### How It Works Internally

Training writes metrics to `mlflow.db` (a SQLite database file). The MLflow UI server reads from this same file via `--backend-store-uri sqlite:///mlflow.db`. The database is gitignored (too large for version control).

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
