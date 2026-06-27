# Training Strategy Analysis: Principled Decisions vs Empirical Tricks

A critical self-assessment of every component in our training pipeline,
examining whether each choice is grounded in published theory or ad-hoc experimentation.

---

## v2 Pipeline Configuration (Current)

```
Model:        ResNet34 U-Net + scSE decoder attention
Loss:         Adaptive Wing Loss (AWL)
Scheduler:    CosineAnnealingWarmRestarts (T_0=20, T_mult=2)
Augmentation: Resize, Rotate±15°, Affine(shift±5%, scale±10%),
              ElasticTransform(α=15), CLAHE, GaussianBlur, GaussNoise,
              RandomBrightnessContrast
Sigma:        Uniform σ=5 for all 25 landmarks
Landmarks:    25 clinically essential (Steiner, Downs, Ricketts, Wits coverage)
```

---

## Component-by-Component Justification

### 1. Adaptive Wing Loss (AWL)

**Citation:** Wang, X., Bo, L., & Fuxin, L. (2019). *Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression.* ICCV 2019.

**Theoretical basis:** In heatmap regression, the target is extremely sparse — at 512×512 with σ=5, only ~0.1% of pixels carry landmark information. Standard MSE averages gradients across all pixels:

$$\nabla_{MSE} = \frac{1}{HW}\sum_{i}^{HW} 2(y_i - \hat{y}_i)$$

This dilutes the useful gradient from ~300 foreground pixels across 262,144 total pixels — a ~870× dilution. AWL addresses this with two mechanisms:

1. **Adaptive curvature:** The loss function's shape adapts based on the ground truth value at each pixel. Near the Gaussian peak (high $y_{target}$), the loss has steeper gradients, forcing precise localisation. In the background (low $y_{target}$), gradients are suppressed.

2. **Foreground weighting:** An explicit weight map multiplies the loss by 11× at foreground pixels.

**Empirical validation from our experiments:**
- Baseline (MSE): 8 of 29 landmarks produced blank heatmaps (model "gave up")
- AWL run: 0 of 25 landmarks produced blank heatmaps

AWL solved the core failure mode. This is not a trick — it's the standard loss function for heatmap regression in face alignment and medical landmark detection.

**Hyperparameters (ω=14, θ=0.5, ε=1, α=2.1):** These are the defaults from the original paper. We did not tune them. A proper ablation should test sensitivity to these values, though the original paper reports robustness across a range.

---

### 2. CosineAnnealingWarmRestarts

**Citation:** Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017.

**Theoretical basis:** ReduceLROnPlateau is reactive — it waits for stagnation, then monotonically decreases LR. Once LR drops, the model cannot escape a local minimum. With 25 output channels, the loss landscape has many more local minima than with 6 channels.

Cosine annealing with warm restarts provides periodic exploration:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_i}\pi))$$

Each restart "kicks" the model out of its current minimum. The cycle length increases (T_mult=2), giving longer convergence windows as training progresses.

**Configuration rationale:**
- `T_0=20`: First restart after 20 epochs — matches the observed early plateau length from baseline
- `T_mult=2`: Cycles of 20 → 40 → 80 → 160 epochs. Each cycle is longer, allowing progressively finer convergence
- `eta_min=1e-6`: Floor prevents gradient underflow

**Empirical validation from our experiments:**
- Baseline (ReduceLROnPlateau): Dead plateau from epoch 10-50 (40 epochs of zero improvement)
- Cosine annealing: Continuous improvement throughout training, no plateau observed

---

### 3. scSE Decoder Attention

**Citation:** Roy, A.G., Navab, N., & Wachinger, C. (2018). *Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks.* MICCAI 2018.

**Theoretical basis:** Standard U-Net skip connections pass ALL encoder features to the decoder without filtering. scSE adds learned spatial and channel attention gates:

- **Channel SE:** Global average pooling → FC layers → per-channel weights. Learns which feature channels (edge detectors, texture filters) are relevant.
- **Spatial SE:** Conv layer → spatial weight map. Learns which spatial locations contain useful features.

This is particularly relevant for multi-landmark detection where different decoder channels need different encoder features for different anatomical regions.

**Implementation:** Single parameter in SMP library (`decoder_attention_type="scse"`). Adds ~5% more parameters. Well-validated in medical image segmentation literature.

---

### 4. Augmentation Pipeline

All augmentations are standard in medical imaging:

| Augmentation | Citation / Standard | Rationale |
|---|---|---|
| **Rotate ±15°** | Universal | Patient head positioning variation |
| **Affine (shift ±5%, scale ±10%)** | Universal | Compensates for varying zoom/positioning |
| **ElasticTransform (α=15)** | Ronneberger et al., 2015 (original U-Net paper) | Simulates anatomical shape variation between patients |
| **CLAHE** | Pizer et al., 1987; standard X-ray preprocessing | Normalises contrast across different X-ray machines (Aariz has 7 machines) |
| **GaussianBlur** | Universal | Robustness to varying image quality |
| **GaussNoise** | Universal | Sensor noise simulation |
| **RandomBrightnessContrast** | Universal | Exposure variation |
| **No horizontal flip** | Domain-specific constraint | Cephalometric landmarks are anatomically asymmetric (left-right has clinical meaning) |

**Why α=15 for elastic (not α=30):** At 512px resolution, dental landmarks are 5-15px apart. Elastic deformation at α=30 can displace pixels by up to ~15px, potentially swapping the relative positions of adjacent dental landmarks. α=15 provides shape variation without disrupting dense landmark topology. This was validated empirically — α=30 correlated with dental landmark collapse in v1.

---

### 5. Uniform Sigma (σ=5)

**What we tried:** Per-landmark sigma (σ=3 dental, σ=5 skeletal, σ=7 soft tissue), hypothesising that tighter Gaussians would reduce overlap in the dense dental region and wider Gaussians would accommodate soft tissue annotation uncertainty.

**Why we reverted:** The interaction between narrow Gaussians (σ=3) and spatial augmentations (Affine shifts, elastic deformation) was destructive. A 3px shift that's tolerable for a σ=5 Gaussian (shift = 0.6σ) becomes significant for σ=3 (shift = 1.0σ), creating noisy gradients that prevent convergence.

**Principled decision:** Without a proper hyperparameter search over sigma values (which would require many training runs), uniform σ=5 is the safest default. The original heatmap regression literature (Newell et al., 2016; Sun et al., 2019) uses uniform sigma.

---

### 6. 25 Landmark Selection

**Rationale:** Selected landmarks required for the three most widely used cephalometric analyses:

- **Steiner analysis:** SNA, SNB, ANB, Go-Gn plane, incisor angulation (U1-NA, L1-NB)
- **Downs analysis:** Frankfort horizontal (Po-Or), facial angle, Y-axis, IMPA
- **Ricketts analysis:** Facial depth, E-plane (lip position), lower face height
- **Wits analysis:** A-point and B-point projected onto functional occlusal plane
- **Soft tissue profile:** Nasolabial angle (Sn), lip position (Ls, Li), nasal projection (Pn)

**Excluded landmarks and justification:**
| Excluded | Reason |
|---|---|
| Ramus (R) | Not used in any standard cephalometric analysis |
| Upper 2nd PM Cusp (UPM) | Research-only landmark, no clinical analysis requires it |
| Lower 2nd PM Cusp (LPM) | Research-only landmark, no clinical analysis requires it |
| Soft Tissue Nasion (N') | Redundant — bony Nasion (N) is detected at 0.87mm and is more clinically important |

This is clinically defensible — a reviewer with orthodontic expertise would agree these 4 are non-essential.

---

## What Was Removed (and Why)

### Online Hard Landmark Mining (OHLM) — Custom, Unvalidated

**What it did:** Dynamically reweighted per-channel loss, giving 0.5-3× weight based on relative channel difficulty.

**Why it was a trick:**
1. No published precedent for per-channel loss reweighting in heatmap regression
2. Loosely inspired by OHEM (Shrivastava et al., 2016) but OHEM operates on *samples*, not *channels* — fundamentally different
3. No theoretical analysis of convergence properties
4. Empirically destructive: channels that started "easy" (dental) were weighted at 0.5×, starving them of gradient signal, leading to collapse

**Lesson:** Custom loss modifications require theoretical justification and careful ablation. Applying an idea from one domain (object detection) to another (heatmap regression) without validation is precisely the kind of "trick" that reviewers reject.

### Adaptive Sigma (σ=3/5/7) — Reasonable Hypothesis, Insufficient Validation

**What it was:** Per-landmark-category sigma values based on anatomical reasoning.

**Why it was removed:**
1. The sigma values (3, 5, 7) were chosen by intuition, not empirical optimisation
2. The interaction with spatial augmentations was not analysed beforehand
3. Published work on heatmap regression universally uses uniform sigma

**Lesson:** Anatomical reasoning can motivate hypotheses, but hypotheses require validation before deployment. This could be revisited with a proper sigma sweep across values.

---

## Experimental Rigour Gaps (To Address for Publication)

### 1. Ablation Study (Required)

We changed multiple variables between runs. A proper ablation isolates each:

| Experiment | Description |
|---|---|
| Baseline | MSE + ReduceLROnPlateau + vanilla U-Net + basic augmentations |
| +AWL only | Replace MSE with AWL, keep everything else from baseline |
| +Cosine only | Replace ReduceLROnPlateau with CosineAnnealing, keep MSE |
| +scSE only | Add scSE attention, keep MSE + ReduceLROnPlateau |
| +Aug only | Enhanced augmentations, keep MSE + ReduceLROnPlateau |
| Full model | AWL + Cosine + scSE + enhanced augmentations |

This table proves which components contribute and by how much. Without it, a reviewer cannot assess the marginal value of each design choice.

### 2. Statistical Significance

Single training runs are anecdotal. For publication:
- Minimum 3 runs with different random seeds
- Report mean ± standard deviation for MRE and SDR
- Statistical tests (paired t-test or Wilcoxon) for comparisons

### 3. Clinical Validation

- **Inter-observer variability:** Compare model error against the disagreement between Junior and Senior orthodontists in the Aariz dataset
- **Clinical angle error propagation:** Translate landmark MRE into angular errors for SNA, SNB, ANB
- **Bland-Altman plots:** Standard clinical agreement visualisation

### 4. Comparison with Published Methods

Must compare against methods evaluated on the same Aariz dataset or similar benchmarks:
- CL-Detection challenge results
- YOLOv12-based approaches
- Other heatmap regression baselines

---

## Summary: Is Our Approach Principled?

| Aspect | Assessment |
|---|---|
| **Architecture (ResNet34 U-Net + scSE)** | ✅ Standard, well-cited |
| **Loss function (AWL)** | ✅ Standard for heatmap regression, peer-reviewed |
| **LR scheduler (Cosine Annealing)** | ✅ Standard, well-cited |
| **Augmentations** | ✅ Standard medical imaging practice |
| **Landmark selection** | ✅ Clinically justified |
| **Sigma choice (uniform 5)** | ✅ Follows published convention |
| **Removed OHLM** | ✅ Correct — was an unvalidated custom modification |
| **Removed adaptive sigma** | ✅ Correct — was an untested hypothesis |
| **Experimental process** | ⚠️ Iterative trial-and-error, needs ablation study for publication |
| **Statistical rigour** | ⚠️ Single runs, no error bars, needs multiple seeds |

**Bottom line:** The v2 configuration is a principled, well-cited training pipeline. Every component has published theoretical and empirical support. The gaps are in experimental methodology (ablation, statistics, clinical validation), not in the technical approach itself.
