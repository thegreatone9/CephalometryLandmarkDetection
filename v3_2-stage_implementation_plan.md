# Two-Stage Cephalometric Landmark Detection Pipeline

Implement a coarse-to-fine pipeline that detects all 29 Aariz landmarks, addressing the channel collapse problem identified in single-stage training.

## Architecture Overview

```
Stage 1 (Coarse):  Full 512×512 image → ResNet34 U-Net → 29 heatmaps → coarse coordinates
                                                            ↓
Stage 2 (Refine):  29 × 96×96 crops around coarse coords → Lightweight CNN → refined coordinates
                                                            ↓
Output:            29 precise landmark coordinates
```

---

## Proposed Changes

### Component 1: Expand to 29 Landmarks

#### [MODIFY] [dataset.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/data/dataset.py)

- Change `SELECTED_SYMBOLS` to use all 29 from `ALL_LANDMARK_SYMBOLS`
- Add display names for the 4 new landmarks: R (Ramus), LPM, UPM, N' (Soft Tissue Nasion)
- Update `SIGMA_MAP` to include entries for R, LPM, UPM, N'
- Update `NUM_LANDMARKS` (now 29)

---

### Component 2: Channel-Balanced Loss

#### [MODIFY] [losses.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/training/losses.py)

Change the final loss reduction from `loss.mean()` (global pixel average) to **per-channel-first averaging**:

```python
# Current (global mean — blank channels are invisible):
return loss.mean()

# Proposed (channel-balanced — each landmark matters equally):
B, C, H, W = loss.shape
per_channel_loss = loss.mean(dim=(0, 2, 3))  # (C,) — mean per channel
return per_channel_loss.mean()               # scalar — mean across channels
```

This ensures every landmark channel contributes equally to the total loss, preventing the "silent student" collapse where blank predictions go unnoticed.

---

### Component 3: Stage 2 Refiner Model

#### [NEW] [refiner.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/models/refiner.py)

A lightweight CNN that takes a small crop and produces a single-channel heatmap:

```python
class LandmarkRefiner(nn.Module):
    """Lightweight refiner for single-landmark heatmap regression.
    
    Takes a 96×96 grayscale crop centered on a coarse prediction
    and outputs a 96×96 heatmap with a Gaussian peak at the 
    refined landmark position.
    """
    def __init__(self, in_channels=1, base_channels=32):
        # 4 conv blocks: 32→64→128→64 with residual connections
        # Final 1×1 conv to 1 channel output
        # ~200K parameters (vs 24M for Stage 1)
```

Key design decisions:
- **Input:** 96×96 grayscale crop (large enough for ~48mm radius at 0.1mm/px)
- **Output:** 96×96 single-channel heatmap
- **Architecture:** Simple CNN with residual blocks — no need for U-Net/encoder-decoder since input is already small
- **Shared model:** One refiner for all 29 landmarks (with landmark identity encoded as a one-hot channel or embedding)

---

### Component 4: Stage 2 Dataset

#### [NEW] [crop_dataset.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/data/crop_dataset.py)

A dataset that:
1. Runs Stage 1 inference to get coarse predictions (or loads pre-computed predictions)
2. Crops a 96×96 patch around each coarse prediction from the original image
3. Generates a ground-truth heatmap for the single landmark within the crop
4. Returns `{"crop": Tensor, "heatmap": Tensor, "landmark_idx": int, "offset": (x, y)}`

The offset records where the crop was taken from, so we can map refined coordinates back to full-image space.

> [!IMPORTANT]
> During training, we add jitter (±10-20px random offset) to the crop centre so the refiner learns to handle imprecise Stage 1 predictions, not just perfectly centred crops.

---

### Component 5: Stage 2 Training

#### [NEW] [train_refiner.py](file:///Users/musakhan/Documents/Practice/cephalometry/train_refiner.py)

Separate training script for Stage 2:

```bash
python train_refiner.py \
    --stage1-checkpoint checkpoints/stage1-29L/best_model.pth \
    --epochs 100 \
    --batch-size 64 \
    --crop-size 96 \
    --jitter 20
```

Flow:
1. Load trained Stage 1 model
2. Run Stage 1 on train/val sets to get coarse predictions
3. Build crop datasets from those predictions
4. Train the refiner with MSE loss (simpler than AWL for single-landmark crops — the foreground/background ratio is much more balanced in a 96×96 crop)
5. Log to MLflow under a separate experiment

> [!NOTE]
> Stage 2 training is fast because: (a) crops are small (96×96 vs 512×512), (b) model is tiny (~200K params), (c) batch size can be large (64+). Expect ~30 minutes on the RTX 3060 Ti.

---

### Component 6: Combined Inference Pipeline

#### [NEW] [pipeline.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/inference/pipeline.py)

End-to-end inference combining both stages:

```python
class TwoStagePipeline:
    def __init__(self, stage1_checkpoint, stage2_checkpoint, device):
        self.stage1 = load_stage1(stage1_checkpoint)
        self.stage2 = load_stage2(stage2_checkpoint)
    
    def predict(self, image: Tensor) -> np.ndarray:
        # 1. Run Stage 1 → coarse coordinates (29, 2)
        # 2. For each landmark, crop 96×96 around coarse prediction
        # 3. Run Stage 2 on each crop → refined coordinates
        # 4. Map refined coordinates back to full image space
        # 5. Return final coordinates (29, 2)
```

#### [MODIFY] [evaluation.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/evaluation.py)

Add `evaluate_pipeline()` that accepts a `TwoStagePipeline` instead of a raw model, handling the two-stage flow transparently.

---

### Component 7: Stage 1 Retraining

#### [MODIFY] [train.py](file:///Users/musakhan/Documents/Practice/cephalometry/train.py)

- Update to use 29 landmarks (via the modified `SELECTED_SYMBOLS`)
- Add `--stage` flag to label this as Stage 1 training
- The rest stays the same — ResNet34 U-Net, AWL (now channel-balanced), cosine annealing

---

## Training Workflow

```
Step 1:  Train Stage 1 on all 29 landmarks (400 epochs, ~11h on RTX 3060 Ti)
         python train.py --epochs 400 --batch-size 12 --img-size 512

Step 2:  Train Stage 2 refiner (100 epochs, ~30min on RTX 3060 Ti)
         python train_refiner.py --stage1-checkpoint checkpoints/.../best_model.pth --epochs 100

Step 3:  Evaluate combined pipeline
         python evaluate_pipeline.py --stage1-ckpt ... --stage2-ckpt ...
```

---

## File Summary

| File | Status | Description |
|------|--------|-------------|
| `src/data/dataset.py` | MODIFY | Expand to 29 landmarks |
| `src/training/losses.py` | MODIFY | Channel-balanced loss reduction |
| `src/models/refiner.py` | NEW | Lightweight Stage 2 CNN |
| `src/data/crop_dataset.py` | NEW | Crop-based dataset for Stage 2 |
| `train_refiner.py` | NEW | Stage 2 training script |
| `src/inference/pipeline.py` | NEW | Combined two-stage inference |
| `src/evaluation.py` | MODIFY | Support pipeline evaluation |
| `train.py` | MODIFY | 29 landmarks, stage label |

---

## Open Questions

> [!IMPORTANT]
> **Refiner architecture — shared or per-landmark?**
> Option A: One shared refiner for all 29 landmarks (simpler, fewer params, but might struggle with diverse landmark types).
> Option B: One refiner per anatomical group (4 models: skeletal, dental, soft tissue, posterior). More capacity but more complexity.
> Recommendation: Start with Option A (shared). If specific landmark groups still fail, split into groups.

> [!IMPORTANT]
> **Crop size — 96 or 128?**
> 96×96 at 0.1mm/px = ~4.8mm radius. Our worst Stage 1 errors are ~70mm (Gonion). If Stage 1 improves with channel-balanced loss (expected), 96 should be sufficient. But if Go/Ar still miss by >48px, we'd need 128.
> Recommendation: Start with 96. If >5% of crops miss the true landmark, increase to 128.

---

## Verification Plan

### Automated Tests
```bash
# After Stage 1 training
python train.py --epochs 400 --batch-size 12

# After Stage 2 training
python train_refiner.py --stage1-checkpoint checkpoints/.../best_model.pth --epochs 100

# Combined evaluation
python -c "from src.inference.pipeline import TwoStagePipeline; ..."
```

### Success Criteria
- **Stage 1:** Overall MRE < 10mm, no landmark > 80mm (coarse predictions stay in the right region)
- **Combined:** Overall MRE < 2.0mm, SDR@2mm > 80%, zero landmarks > 10mm
- **Per-landmark:** All 29 landmarks < 4mm MRE (clinically acceptable)
