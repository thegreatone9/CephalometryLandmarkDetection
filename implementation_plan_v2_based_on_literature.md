# Architecture Improvement Plan v2

Grounded in literature analysis and diagnosis of our current training behavior.
Every change has a **why** before the **what**.

---

## Diagnosis: Why Our Current Model Struggles on 29 Landmarks

From the epoch 50 evaluation, we observed:
- **22 of 29 landmarks** had predictions collapsed to a mean location (~400, 370)
- **Only 3 landmarks** (LIT, UIT, B-point) showed reasonable accuracy
- **Sella and Nasion** — previously our best landmarks at 6-channel — are now 279px and 495px off

### Root Causes (not symptoms)

```
Root Cause 1: MSE Loss Blindness
├── 99.88% of target heatmap pixels are zero
├── Model gets "free" loss reduction from predicting zeros everywhere
├── Gradient signal from the landmark bump is drowned out
└── Result: Model defaults to a safe "mean" prediction rather than risking
    a wrong bump location (which would increase MSE MORE than just zeros)

Root Cause 2: No LR Exploration
├── ReduceLROnPlateau with patience=10 only reacts AFTER stagnation
├── It reduces LR monotonically — never increases it back
├── Once LR drops, the model can't escape local minima
└── Result: Epoch 10-50 plateau (the model was stuck, not converging)

Root Cause 3: Gaussian Overlap in Dental Region
├── At 512px, dental landmarks (UIT/LIT/UIA/LIA/UPM/LPM) are 5-15px apart
├── With σ=5, each Gaussian bump spans ~25px (5σ diameter)
├── Adjacent dental landmark Gaussians OVERLAP significantly
├── The model cannot learn to distinguish them — the supervision is ambiguous
└── Result: Dental landmarks collapse to a shared location

Root Cause 4: Insufficient Augmentation Diversity
├── Only rotation ±15°, brightness/contrast, and Gaussian noise
├── No scale variation (model only sees exactly 512px crops)
├── No elastic deformation (anatomical shape variation)
├── No CLAHE (contrast enhancement that mimics different X-ray machines)
└── Result: Model overfits to the specific intensity/scale distribution
```

---

## Change 1: Adaptive Wing Loss (AWL)

### The Principle

MSE treats all pixels equally: $\mathcal{L}_{MSE} = \frac{1}{N}\sum(y - \hat{y})^2$

But in a 512×512 heatmap with σ=5 Gaussian, only ~300 of 262,144 pixels carry landmark information. MSE averages the gradient across all pixels, so the useful gradient from those 300 pixels is diluted by a factor of ~870×.

Adaptive Wing Loss (Wang et al., ICCV 2019) solves this with two mechanisms:

**1. Non-linear loss shape** — For small errors (near the Gaussian peak), AWL has a higher gradient than MSE, forcing the model to refine precise positioning:

$$AWing(x) = \begin{cases} \omega \ln(1 + |x/\epsilon|^{\alpha-y}) & \text{if } |x| < \theta \\ A|x| - C & \text{otherwise} \end{cases}$$

where $\omega, \epsilon, \theta, \alpha$ are hyperparameters, and **$y$ is the ground truth value at that pixel**.

The key insight: when $y$ is high (foreground pixel near landmark), the loss function becomes more curved — small errors get large gradients. When $y$ is low (background), the loss flattens out — background errors are de-emphasized.

**2. Weighted loss map** — Multiplies the loss by a mask $W$ that gives:
- Weight = 1.0 for foreground pixels (within the Gaussian)
- Weight = higher for "hard negative" pixels (near but not on the landmark)
- Weight = low for far-background pixels

### What This Fixes

| Problem | How AWL fixes it |
|---|---|
| Gradient dilution from 99.9% zeros | Background pixels generate near-zero gradients |
| Model predicts "safe mean" location | Wrong foreground predictions get MUCH higher gradients than background errors |
| Slow convergence on hard landmarks | Small positioning errors near peaks get amplified gradients for refinement |

### Implementation

#### [NEW] `src/training/losses.py`

```python
class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        """
        omega: amplification factor for foreground
        theta: threshold switching linear/nonlinear regime
        epsilon: prevents division by zero
        alpha: shape parameter (must be > y for foreground)
        """
        
    def forward(self, pred, target):
        # Compute AWL per pixel with target-adaptive shape
        # Apply weighted loss map (higher weight near landmarks)
```

#### [MODIFY] `src/training/trainer.py` line 119

```diff
- self.criterion = nn.MSELoss()
+ from src.training.losses import AdaptiveWingLoss
+ self.criterion = AdaptiveWingLoss(omega=14, theta=0.5, epsilon=1, alpha=2.1)
```

**Effort:** ~50 lines of code, 1 hour

---

## Change 2: Cosine Annealing with Warm Restarts

### The Principle

`ReduceLROnPlateau(patience=10, factor=0.5)` is **reactive** — it waits for 10 epochs of stagnation, then cuts LR by half. It never increases LR. This creates a one-way path: LR can only decrease over time.

The problem: once LR drops, the model can't escape a local minimum. It's stuck making tiny updates that barely change the loss. This explains our epoch 10-50 plateau.

**Cosine Annealing with Warm Restarts** (Loshchilov & Hutter, ICLR 2017) is **proactive**:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_i}\pi))$$

It **periodically resets** the learning rate back to a high value, creating exploration cycles:

```
LR: ▄▃▂▁▄▃▂▁▄▃▂▁  (cosine with restarts)
vs
LR: ▄▄▄▄▃▃▃▃▂▂▁▁  (ReduceLROnPlateau - monotonically decreasing)
```

Each restart "kicks" the model out of its current minimum and forces it to explore the loss landscape. The cosine curve ensures smooth transitions — no sudden jumps.

### Why This Matters for 29 Landmarks

With 29 output channels, the loss landscape has many more local minima than with 6. The model needs periodic exploration to find configurations where ALL 29 channels are well-placed, not just the easiest few. ReduceLROnPlateau found a minimum where ~3 landmarks were good and got stuck there.

### Implementation

#### [MODIFY] `src/training/trainer.py` lines 110-116

```diff
- from torch.optim.lr_scheduler import ReduceLROnPlateau
+ from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

- self.scheduler = ReduceLROnPlateau(
-     self.optimizer, mode="min", factor=0.5, patience=10,
- )
+ self.scheduler = CosineAnnealingWarmRestarts(
+     self.optimizer,
+     T_0=20,        # First restart after 20 epochs
+     T_mult=2,      # Each subsequent cycle is 2× longer (20, 40, 80, ...)
+     eta_min=1e-6,  # Floor LR
+ )
```

Also need to change the scheduler step call — `CosineAnnealingWarmRestarts` steps per epoch (not per validation loss):

```diff
- self.scheduler.step(val_loss)  # reactive
+ self.scheduler.step()          # proactive, per epoch
```

**Why T_0=20, T_mult=2:** 
- First cycle (epochs 1-20): Fast exploration, model learns coarse landmark positions
- Second cycle (epochs 20-60): Longer refinement, model locks in moderate landmarks  
- Third cycle (epochs 60-200): Long convergence, hard landmarks (soft tissue, dental) converge

**Effort:** ~5 lines changed, 15 minutes

---

## Change 3: Category-Aware Adaptive Sigma

### The Principle

Currently, all 29 landmarks use σ=5. But landmarks have fundamentally different spatial characteristics:

**Skeletal landmarks** (Sella, Nasion, Menton) sit on clear bony edges with high contrast. They're anatomically isolated — the nearest other landmark is typically 50+ pixels away at 512px. σ=5 works fine.

**Dental landmarks** (UIT, LIT, UIA, LIA, UPM, LPM, UMT, LMT) are clustered in a small region of the jaw. At 512px, neighboring dental landmarks may be only 5-15 pixels apart. With σ=5, their Gaussians overlap by 50%+:

```
At 512px, UIT and LIT might be 8px apart.
Gaussian with σ=5 has 95% of its mass within ±10px.
So UIT's Gaussian significantly overlaps LIT's Gaussian.
The model sees ambiguous supervision: "put a bump somewhere around here."
```

**Soft tissue landmarks** (Pronasale, Subnasale, Labrale) have inherently higher annotation uncertainty because soft tissue boundaries are fuzzy. A larger σ reflects this uncertainty honestly and prevents the model from chasing noise in the annotations.

### The Fix

```python
SIGMA_MAP = {
    # Skeletal — clear bony edges, well-separated
    "S": 5.0, "N": 5.0, "A": 5.0, "B": 5.0, "Pog": 5.0, "Me": 5.0,
    "Gn": 5.0, "Go": 5.0, "Ar": 5.0, "Co": 5.0, "Or": 5.0,
    "ANS": 5.0, "PNS": 5.0, "R": 5.0, "Po": 5.0,
    
    # Dental — tightly clustered, need narrow Gaussians
    "UIT": 3.0, "UIA": 3.0, "LIT": 3.0, "LIA": 3.0,
    "UMT": 3.0, "LMT": 3.0, "UPM": 3.0, "LPM": 3.0,
    
    # Soft tissue — fuzzy boundaries, higher annotation uncertainty
    "Pn": 7.0, "Sn": 7.0, "Ls": 7.0, "Li": 7.0,
    "N`": 7.0, "Pog`": 7.0,
}
```

### Implementation

#### [MODIFY] `src/data/dataset.py` — `generate_heatmaps` function

Currently takes a single `sigma` parameter. Modify to accept a list of per-landmark sigmas:

```diff
- def generate_heatmaps(landmarks, height, width, sigma=5.0):
+ def generate_heatmaps(landmarks, height, width, sigma=5.0, sigma_per_landmark=None):
      # For each landmark i:
-     # Use sigma
+     # Use sigma_per_landmark[i] if provided, else sigma
```

#### [MODIFY] `src/data/dataset.py` — `CephalometricDataset.__getitem__`

```diff
  heatmaps = generate_heatmaps(
-     landmarks_out, h_out, w_out, sigma=self.sigma,
+     landmarks_out, h_out, w_out, sigma=self.sigma,
+     sigma_per_landmark=self.sigma_per_landmark,
  )
```

**Effort:** ~30 lines of code, 30 minutes

---

## Change 4: Stronger Augmentation Pipeline

### The Principle

Our current augmentations are minimal:
```python
A.Rotate(limit=15, p=0.5)
A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5)
A.GaussNoise(std_range=(0.01, 0.05), p=0.3)
```

This is insufficient for 29 landmarks because:

1. **No scale variation** — The model only sees exactly 512×512 crops. Real X-rays have varying zoom levels and patient sizes. Without scale augmentation, the model can't generalize to different-sized anatomy.

2. **No elastic deformation** — Anatomical structures vary between patients (jaw angles, skull shapes). Elastic deformation simulates this variation.

3. **No CLAHE** — Different X-ray machines produce different contrast distributions. CLAHE (Contrast Limited Adaptive Histogram Equalization) is a standard X-ray preprocessing technique that the CL-Detection challenge winner used.

4. **No shift** — Without translation augmentation, the model expects landmarks at fixed absolute positions. This may explain the "collapse to mean" behavior.

### The Fix

```python
def get_train_transforms(image_size=512):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        
        # Geometric — simulate patient positioning variation
        A.Rotate(limit=15, border_mode=0, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,    # ±5% translation
            scale_limit=0.10,    # ±10% zoom
            rotate_limit=0,      # rotation handled above
            border_mode=0, p=0.5,
        ),
        
        # Elastic — simulate anatomical variation
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),
        
        # Intensity — simulate different X-ray machines
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.5,
        ),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
```

### Why Each Addition

| Augmentation | Simulates | Impact |
|---|---|---|
| **ShiftScaleRotate** | Patient positioning variation, zoom | Breaks positional bias (prevents "collapse to mean") |
| **ElasticTransform** | Anatomical shape variation | Better generalization across jaw shapes |
| **CLAHE** | Different X-ray machine contrast | Cross-device robustness (our Aariz data has 7 machines) |
| **GaussianBlur** | Varying image quality / focal blur | Robustness to image quality |

**Effort:** ~10 lines changed, 15 minutes

---

## Change 5: SMP Decoder Attention

### The Principle

The vanilla U-Net skip connections pass ALL encoder features to the decoder without filtering. For 29 landmarks, this creates noise — features relevant for Sella (top of skull) are irrelevant for Menton (bottom of chin), but they're all passed through the same skip connections.

**Attention gates** (Oktay et al., 2018) add a learned filter at each skip connection. The decoder's high-level features (which know roughly WHERE the model is looking) gate the encoder's low-level features (which have the fine spatial detail). Only spatially relevant features pass through.

`segmentation_models_pytorch` supports this natively via the `decoder_attention_type` parameter:

```python
smp.Unet(
    encoder_name="resnet34",
    decoder_attention_type="scse",  # Spatial + Channel Squeeze-Excitation
    ...
)
```

**scSE (Spatial and Channel Squeeze-Excitation)** combines:
- **Channel SE**: learns which feature channels are important (e.g., "edge features" vs "texture features")
- **Spatial SE**: learns which spatial locations are important (e.g., "jaw region" vs "skull base")

### Implementation

#### [MODIFY] `src/models/unet.py` line 41-46

```diff
  model = smp.Unet(
      encoder_name=encoder_name,
      encoder_weights=encoder_weights,
      in_channels=in_channels,
      classes=num_classes,
+     decoder_attention_type="scse",
  )
```

**Effort:** 1 line, 1 minute. But adds ~5% more parameters and improves feature selection at every decoder level.

---

## Implementation Order and Rationale

These changes are ordered by **independence** (no dependencies between them) and **impact per effort**:

| Phase | Change | Effort | Expected Impact | Rationale |
|---|---|---|---|---|
| **Phase 1** | AWL Loss + Cosine LR | 1.5h | ⭐⭐⭐⭐⭐ | Fixes the two fundamental training dynamics issues. These are the reason most landmarks aren't converging. |
| **Phase 2** | scSE Attention | 1 min | ⭐⭐⭐ | Free improvement — one line, better feature selection. |
| **Phase 3** | Adaptive Sigma | 30 min | ⭐⭐⭐⭐ | Fixes dental landmark overlap — currently unsolvable by the model. |
| **Phase 4** | Stronger Augmentation | 15 min | ⭐⭐⭐ | Prevents overfitting and improves cross-device generalization. |

**Total implementation time: ~2.5 hours**

### Training Plan for Tomorrow

```bash
# Run 1: All improvements, same config as baseline for fair comparison
python train.py --encoder resnet34 --epochs 200 --batch-size 8 --img-size 512

# Run 2 (if time permits): Higher resolution ablation  
python train.py --encoder resnet34 --epochs 200 --batch-size 4 --img-size 640
```

Compare Run 1 vs tonight's baseline to quantify the improvement. This comparison becomes a section in the paper: **"Ablation Study: Effect of Training Strategy on Detection Accuracy."**

---

## What We're NOT Changing (and Why)

| Technique from literature | Why we skip it |
|---|---|
| **Dual-encoder coarse→fine (D-CeLR)** | Requires completely new architecture. Our U-Net is already working — we need to fix training, not rebuild from scratch. This would be a new paper, not an improvement. |
| **ConvNeXt V2 encoder** | Diminishing returns. ResNet34 is well-validated for this task. Encoder choice matters less than loss function and training strategy. |
| **Graph neural networks for landmark relationships** | Over-engineering. Useful when you have 100+ landmarks. For 29, spatial attention in the decoder is sufficient. |
| **Diffusion-based data augmentation** | Separate research project. Standard augmentations + CLAHE address the core diversity issue. |
| **Deep supervision** | Adds complexity with marginal gains when AWL already fixes the gradient signal issue. Could revisit if results are still unsatisfactory. |
