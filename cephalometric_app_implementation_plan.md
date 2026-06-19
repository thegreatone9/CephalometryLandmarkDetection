# Cephalometric Landmark Detection — Implementation Plan

## Project Goal
Build a working system that takes a lateral cephalogram (skull X-ray) as input and automatically detects 6 key anatomical landmarks, then computes the ANB angle (a standard orthodontic measurement) from those landmarks. The output should visually overlay the detected landmarks on the X-ray and display the computed angle in a way a non-specialist can understand.

This is a supervisor-facing demo, but it must be built as a real, runnable application — not a notebook. Prioritize a working end-to-end pipeline over architectural sophistication, but build it with the project structure, packaging, and extensibility of a small real product, since this is also the starting point for an eventual deployable web app.

## Architecture Overview

The project should be structured as a standalone Python application with clearly separated concerns, so it can grow beyond this demo without rewrites:

```
cephalometric-demo/
├── data/                  # dataset download/cache location (gitignored)
├── src/
│   ├── data/              # dataset loading, preprocessing, augmentation
│   ├── models/            # model architecture definitions
│   ├── training/          # training loop, MLflow logging
│   ├── inference/         # inference + post-processing (heatmap → coords → angles)
│   └── viz/                # visualization/overlay generation
├── app/
│   └── streamlit_app.py   # Streamlit UI, imports from src/
├── checkpoints/           # saved model weights (gitignored, mounted in Docker)
├── mlruns/                 # MLflow tracking data (gitignored, mounted in Docker)
├── Dockerfile
├── requirements.txt
├── train.py                # CLI entrypoint for training
└── README.md
```

The Streamlit app must import shared logic from `src/` rather than duplicating preprocessing or inference code — this is what makes the demo extensible into a real web app later (e.g. swapping Streamlit for a FastAPI backend without touching the model/inference logic).

---

## Dataset

**Primary dataset: ISBI 2015 Cephalometric Challenge Dataset**
- Source: https://www.kaggle.com/datasets/soumikrakshit/isbi-challenge-dataset
- 400 lateral cephalometric X-ray images, TIFF format, ~1935×2400 resolution
- 19 landmarks per image, annotated by two expert clinicians (use the average or the junior annotator's labels as ground truth — be consistent)
- Official split: 150 training, 150 Test1, 100 Test2 (combine Test1+Test2 as a held-out test set, or follow whatever split convention is documented with the data)

**Do not** merge in any additional datasets (e.g. Cephalometrix) for this phase. Single dataset, single device, single format — this is a deliberate scope decision to keep the pipeline simple. Multi-dataset merging is a planned future extension, not part of this build.

### Landmark Subset
From the 19 available landmarks, select only these 6 for this demo:
1. Sella (S)
2. Nasion (N)
3. A-point (A)
4. B-point (B)
5. Pogonion (Pog)
6. Menton (Me)

Confirm these exact 6 names/indices against the dataset's annotation key/order before writing the data loader — annotation files typically list landmarks in a fixed order; identify the correct indices for these 6 and extract only those.

---

## Step 1: Data Pipeline

Build a dataset loader that:
1. Reads each X-ray image and its corresponding landmark coordinate file
2. Resizes all images to a consistent input size (use 512×512 or 800×640 — pick one and apply it consistently to both images and landmark coordinates, scaling landmark coordinates proportionally to match the resize)
3. Converts grayscale TIFF to a single-channel tensor, normalized appropriately
4. For each of the 6 selected landmarks, generates a target heatmap: a 2D Gaussian centered on the true (x, y) landmark position, on a blank canvas the same size as the resized image. Use a Gaussian sigma in the range of 3-6 pixels (tune based on visual inspection of heatmap sharpness vs. coverage)
5. Output: input image tensor of shape `[1, H, W]`, target tensor of shape `[6, H, W]` (one heatmap channel per landmark)

Implement data augmentation (apply to both image and landmark coordinates together, keeping them in sync):
- Random rotation: ±15 degrees
- Random brightness/contrast jitter
- Gaussian noise injection
- Do NOT apply horizontal flip (landmarks are anatomically asymmetric — flipping would require remapping left/right landmark identities, which adds unnecessary complexity for this scope)

Split data into train/validation/test sets, ensuring a representative spread (do not let any one subset skew significantly in image characteristics if the official split groups by collection batch).

---

## Step 2: Model Architecture

Use the `segmentation_models_pytorch` library.

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=6,
)
```

- Encoder: ResNet34, pretrained on ImageNet (do NOT freeze/lock the encoder — fine-tune the entire network end-to-end)
- Decoder: standard U-Net decoder, trained from scratch
- Output: 6-channel heatmap prediction matching the target shape from Step 1

Use differential learning rates:
- Encoder (ResNet34) parameters: learning rate ~1e-4
- Decoder parameters: learning rate ~1e-3

This requires setting up parameter groups in the optimizer rather than a single uniform learning rate.

---

## Step 3: Training Loop

- Loss function: MSE (Mean Squared Error) between predicted and target heatmaps. If results are unsatisfying, consider switching to a weighted MSE that penalizes errors near the peak more heavily, but start with plain MSE.
- Optimizer: Adam, with the differential learning rate parameter groups described above
- Batch size: as large as available GPU memory allows (likely small, e.g. 4-8, given image size — adjust based on actual hardware)
- Epochs: start with 50-100, monitor validation loss for plateau/overfitting, adjust accordingly
- Learning rate scheduling: reduce on plateau (monitor validation loss)
- Checkpointing: save the best model based on validation loss
- Log training/validation loss per epoch for review

---

## Step 4: Coordinate Extraction (Inference Post-Processing)

After the model outputs 6 predicted heatmaps for a given image:
1. For each of the 6 channels, find the pixel location of the maximum value (argmax) — this is the predicted landmark coordinate
2. Optionally, for more precision, compute a weighted centroid using a small window around the peak rather than a single argmax pixel
3. Rescale the extracted coordinates back to the original image resolution if working with a resized image

---

## Step 5: Clinical Measurement — ANB Angle Calculation

Using three of the six detected landmarks — **Sella (S), Nasion (N), A-point (A), B-point (B)** — compute:

- **SNA angle**: the angle at vertex N, between rays N→S and N→A
- **SNB angle**: the angle at vertex N, between rays N→S and N→B
- **ANB angle**: SNA − SNB

Implement this as a standard geometry function: given three 2D points, compute the angle at the shared vertex using vector dot product / `atan2`. Display the resulting ANB value alongside the visualization.

(Reference ranges for context, not required to implement logic around: ANB ~2° is considered skeletal Class I/normal; significantly higher suggests Class II; significantly negative suggests Class III. This context is for the demo narrative, not a classification feature to build.)

---

## Step 6: Visualization Output

Produce a visualization function that:
1. Displays the original X-ray image
2. Overlays the 6 predicted landmark points (as small markers, e.g. colored dots or crosses)
3. Optionally overlays the 6 ground-truth points in a different color/style for visual comparison (useful for the demo to show accuracy intuitively)
4. Draws the lines used in the ANB calculation (S-N, N-A, N-B) for visual clarity
5. Displays the computed SNA, SNB, and ANB values as text on or near the image

This is the core demo artifact — prioritize clarity and visual polish here, since this is what gets shown to the supervisor.

---

## Step 7: Evaluation Metrics

Report on the held-out test set:
- **Mean Radial Error (MRE)**: average Euclidean distance (in mm, using the dataset's known pixel spacing of 0.1mm/pixel) between predicted and ground-truth landmarks, per landmark and averaged across all 6
- **Successful Detection Rate (SDR)** at thresholds of 2mm, 2.5mm, 3mm, 4mm — percentage of predictions falling within each distance threshold of ground truth

These are the standard metrics used in the cephalometric landmark detection literature — use them as-is for comparability and credibility in the demo.

---

## Step 8: Streamlit Application

Build `app/streamlit_app.py` as the primary demo interface. No Jupyter/Colab notebooks anywhere in the project — all code lives in proper `.py` modules under `src/`, imported by the app and by `train.py`.

### Upload Flow
The app must let a user upload their own lateral cephalogram image (support common formats: PNG, JPG, TIFF). Since the actual end users of this demo (e.g. a non-dental supervisor) won't have a personal X-ray to upload, also provide:
- A small set of bundled sample X-rays (a few held-out test images bundled with the app) selectable via a dropdown or gallery, so anyone can run the demo with one click without needing their own file
- Clear on-screen guidance on what kind of image is expected (a side-profile/lateral skull X-ray, not a regular dental photo or front-facing scan) so the upload path doesn't silently fail or produce nonsense on the wrong image type
- Basic input validation: confirm the uploaded file is a readable image of reasonable dimensions before running inference; show a friendly error message if not

### Prediction Display (for a non-specialist audience)
After inference, the app must show, in plain language:
- The X-ray with the 6 predicted landmarks overlaid (from Step 6), plus the S-N, N-A, N-B lines used in the angle calculation
- The computed SNA, SNB, and ANB angle values
- A short, plain-English interpretation of what ANB means and what the computed value suggests — e.g. "ANB measures how the upper and lower jaw line up relative to each other. A value around 2° is typical; this patient's value of X° suggests [normal alignment / upper jaw positioned forward / lower jaw positioned forward], which is the kind of measurement an orthodontist uses to decide on a treatment approach." Avoid clinical jargon beyond what's defined inline.
- A one-paragraph explanation of how the prediction was made, in accessible terms — e.g. "The model was trained on X-ray images where doctors had marked these points by hand, and learned to recognize the same anatomical features. It outputs a confidence map for each landmark and picks the most likely point." This is the "explain it to someone outside the field" requirement — write this copy carefully, it's a core deliverable, not a footnote.

### Confidence Scores
For each of the 6 landmarks, display a confidence score derived from the predicted heatmap (e.g. the peak heatmap activation value, normalized to a 0-100% scale, or the inverse of the heatmap's spread/entropy as a sharper proxy for confidence — pick one approach and document why in the README). Display these as a simple table or set of labeled bars next to the landmark names, sorted with the top-confidence landmarks first ("top confidence scores"). If any landmark falls below a reasonable confidence threshold, visually flag it (e.g. a warning icon) so the user understands the prediction may be less reliable for that point — this also reinforces the "this is assistive, not infallible" framing that matters for a clinical-adjacent tool.

### Model Selection (Extensibility Hook)
Since multiple training runs will be tracked via MLflow (see below), the Streamlit app should allow selecting which trained model/checkpoint to run inference with (e.g. a dropdown listing available checkpoints in the `checkpoints/` directory, labeled with their MLflow run name or key metric). This doesn't need to be elaborate — a simple selectbox is sufficient — but it must not hardcode a single model path, since comparing runs is part of the brief.

---

## Step 9: MLflow Experiment Tracking

Integrate MLflow into the training pipeline (`src/training/` and `train.py`) to track multiple training runs:

- Log hyperparameters per run: encoder backbone choice, learning rates (encoder/decoder), batch size, number of epochs, heatmap sigma, augmentation settings
- Log metrics per epoch: training loss, validation loss
- Log final evaluation metrics (from Step 7): MRE (overall and per-landmark), SDR at each threshold (2mm, 2.5mm, 3mm, 4mm)
- Log model artifacts: save the trained model checkpoint as an MLflow artifact, in addition to (or instead of) ad hoc checkpoint files, so each run's model is retrievable directly from MLflow
- Use a consistent experiment name (e.g. `cephalometric-landmark-detection`) so all runs are comparable side by side in the MLflow UI
- Tag each run descriptively (e.g. with the encoder backbone name) to make runs easy to distinguish when comparing

This should be set up so that running `train.py` with different arguments (e.g. different encoder backbones, learning rates, or augmentation settings) automatically produces a new tracked run, with no manual bookkeeping required. The README must explain how to launch the MLflow UI (`mlflow ui`) to browse and compare these runs.

This is also where the "different strategies" mentioned in scope should live: at minimum, track one run using the plain U-Net + ResNet34 setup from Step 2, and structure the code so a second run (e.g. a different backbone, or a different heatmap sigma, or with/without certain augmentations) can be launched as a comparison point. You don't need to exhaustively explore strategies — the goal is that the MLflow setup makes comparing whatever strategies you do try straightforward and visible.

---

## Step 10: Dockerization

Provide a `Dockerfile` at the project root that:
- Uses an appropriate Python base image (e.g. `python:3.10-slim`)
- Installs system dependencies needed by image libraries (e.g. `libgl1` for OpenCV, if used)
- Copies `requirements.txt` and installs Python dependencies
- Copies the project source (`src/`, `app/`, bundled sample images, and a pretrained checkpoint or a way to fetch one)
- Exposes the Streamlit default port (8501)
- Runs the app via `streamlit run app/streamlit_app.py --server.address=0.0.0.0`

The application must be runnable with exactly:
```bash
docker build -t cephalometric-demo .
docker run -p 8501:8501 cephalometric-demo
```
with no additional manual setup steps required after these two commands. This means a trained model checkpoint must either be baked into the image (acceptable for this demo's scope, given it's a small model) or fetched automatically at container startup — do not require the user to manually copy files into the container.

If MLflow tracking data needs to persist outside the container for inspection, document the appropriate volume mount (e.g. `-v $(pwd)/mlruns:/app/mlruns`) as an optional addition in the README, but the core `docker build` + `docker run` flow must work for the Streamlit demo without it.

---

## Step 11: README

Write a `README.md` that clearly explains, in order:

1. **Project overview** — the clinical problem and what this system does, in the same plain-language framing used in the Streamlit app itself
2. **Technical approach** — the full pipeline: dataset used, the 6 selected landmarks and why, preprocessing steps (resizing, normalization, heatmap target generation), augmentation strategy, and why each choice was made (e.g. why heatmap regression over direct coordinate regression, why a pretrained ResNet34 backbone, why differential learning rates)
3. **Model training** — how to run `train.py`, what arguments/configuration are available, and what the training loop does
4. **MLflow tracking** — how runs are logged, what's tracked, how to launch the MLflow UI, and how to compare runs/strategies
5. **Evaluation** — what MRE and SDR mean and how to interpret them, with the project's own results once available
6. **Docker usage** — the exact `docker build` / `docker run` commands, what they do, and any optional flags (e.g. volume mounts)
7. **Running the Streamlit app** — both via Docker and locally (e.g. `streamlit run app/streamlit_app.py`) for development
8. **Project structure** — a short explanation of the folder layout and why it's organized this way (separation of `src/` logic from the `app/` UI, to support future extensibility, e.g. replacing Streamlit with another frontend or deploying the inference logic as an API)
9. **Limitations and scope** — explicitly state this is a demo/prototype, not a validated clinical tool, trained on a single public dataset with 6 of the standard 19 landmarks, and list the deliberate out-of-scope items below as known next steps

The README is a required deliverable, not optional documentation — write it as carefully as the code, since it's part of what gets evaluated.

---

## Explicit Out-of-Scope (Do Not Implement)

- Jupyter/Colab notebooks as the primary deliverable — all code must be in proper `.py` modules; a notebook may exist only as an optional exploratory scratchpad outside the main app/training path, never as the way to run the project
- Autoencoder/self-supervised pretraining on unlabeled X-rays
- Frozen/locked encoder during fine-tuning
- HRNet or any architecture beyond the specified U-Net + ResNet34 (a second backbone may be tried as one of the MLflow-tracked comparison strategies, but is not required)
- Multi-dataset merging (Cephalometrix, CL-Detection, etc.)
- All 19 landmarks — only the 6 specified
- Mobile deployment, DICOM handling, or any native mobile/production-grade deployment concerns beyond the Streamlit + Docker setup specified here
- Treatment classification (Class I/II/III) as a model output — angle computation and plain-language interpretation only, no classification head
- User authentication, multi-user support, or persistent storage of uploaded patient images — this is a stateless single-session demo
- Kubernetes, cloud deployment, or orchestration beyond a single Docker container — those remain explicit future extensions, not part of this build

These are noted as deliberate future extensions, not omissions to fix now.
