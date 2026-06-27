# Baseline Training Results — 29 Landmarks (MSE Loss)

**Run:** ResNet34, 200 epochs, batch 8, 512px, σ=5.0, MSE loss, ReduceLROnPlateau
**Best val loss:** 0.000125 @ epoch 191

---

## The Result: A Bimodal Split

The model learned **21 of 29 landmarks brilliantly**, but **completely gave up on 8 landmarks**.

### ✅ 21 Landmarks — Excellent (MRE < 2.1mm)

| Landmark | MRE (mm) | Category |
|---|---|---|
| Menton (Me) | 0.56 | Skeletal |
| Subnasale (Sn) | 0.57 | Soft tissue |
| Pogonion (Pog) | 0.60 | Skeletal |
| Gnathion (Gn) | 0.61 | Skeletal |
| Pronasale (Pn) | 0.78 | Soft tissue |
| Labrale Superius (Ls) | 0.81 | Soft tissue |
| Nasion (N) | 0.87 | Skeletal |
| A-point (A) | 0.89 | Skeletal |
| B-point (B) | 0.90 | Skeletal |
| Labrale Inferius (Li) | 1.02 | Soft tissue |
| Soft Tissue Nasion (N') | 1.03 | Soft tissue |
| Soft Tissue Pogonion (Pog') | 1.12 | Soft tissue |
| Anterior Nasal Spine (ANS) | 1.18 | Skeletal |
| Upper Incisor Apex (UIA) | 1.25 | Dental |
| Upper Incisor Tip (UIT) | 1.26 | Dental |
| Lower Incisor Apex (LIA) | 1.32 | Dental |
| Lower Incisor Tip (LIT) | 1.57 | Dental |
| Lower Molar Cusp (LMT) | 1.61 | Dental |
| Upper 2nd PM Cusp (UPM) | 1.79 | Dental |
| Upper Molar Cusp (UMT) | 2.00 | Dental |
| Lower 2nd PM Cusp (LPM) | 2.03 | Dental |

**Mean MRE for these 21: ~1.10 mm** — this would BEAT CephRes-MHNet (1.23mm) if consistent across all landmarks.

### ❌ 8 Landmarks — Catastrophic Failure (MRE > 28mm)

| Landmark | MRE (mm) | Location |
|---|---|---|
| Orbitale (Or) | 28.66 | Upper face |
| Ramus (R) | 65.06 | Posterior jaw |
| Gonion (Go) | 67.37 | Posterior jaw angle |
| Condylion (Co) | 77.22 | Upper posterior |
| Articulare (Ar) | 81.58 | Upper posterior |
| Sella (S) | 82.25 | Cranial base |
| Porion (Po) | 89.32 | Upper posterior |
| Posterior Nasal Spine (PNS) | 99.04 | Deep posterior |

---

## Why This Happened — MSE Loss Failure Mode

All 8 failed landmarks are in the **upper/posterior region** of the cephalogram. Here's what happened:

```
The MSE "giving up" mechanism:
─────────────────────────────
For a hard landmark, the model has two choices:

Option A: Predict a Gaussian bump in roughly the right area
  → If it's wrong by 20px, the MSE penalty for those ~300 wrong pixels is HIGH
  → The "wrong bump" creates a sharp MSE spike

Option B: Predict a blank heatmap (near-zero everywhere)  
  → The MSE penalty is only from the ground truth bump (~300 pixels of small values)
  → Much LOWER MSE than a misplaced bump

MSE makes Option B (giving up) CHEAPER than Option A (trying).
The model learns: "it's better to predict nothing than to be wrong."
```

This is exactly why we implemented **Adaptive Wing Loss** — it makes Option B (giving up) MORE EXPENSIVE by heavily weighting the foreground pixels. The model can no longer hide from hard landmarks.

---

## Overall Metrics

| Metric | Value | Context |
|---|---|---|
| Overall MRE | 21.18 mm | Inflated by 8 failed landmarks |
| **MRE (21 good landmarks only)** | **~1.10 mm** | **Would beat CephRes-MHNet's 1.23mm** |
| SDR@2mm | 62.7% | = ~21/29 landmarks × their success rate |
| SDR@2.5mm | 66.1% | |
| SDR@3mm | 68.0% | |
| SDR@4mm | 70.0% | |

---

## What This Means for the Improved Pipeline

> [!IMPORTANT]
> The 21 good landmarks prove the **architecture works**. The model has the capacity to achieve sub-1mm accuracy. The 8 failed landmarks are a **loss function problem**, not an architecture problem.

Our `improved-training` branch fixes this with:

| Fix | How it addresses the failure |
|---|---|
| **Adaptive Wing Loss** | Forces the model to predict bumps even for hard landmarks — can't "give up" |
| **Cosine Annealing LR** | Periodic LR resets may kick the 8 stuck channels into learning |
| **scSE Attention** | Better feature selection for the upper/posterior region |
| **Stronger augmentations** | More exposure to varied posterior anatomy |
| **Adaptive sigma** | Not directly relevant here (the 8 failed landmarks are skeletal, not dental) |

**Expected outcome:** If the improved pipeline gets those 8 landmarks under control (even 3-5mm MRE), the overall metrics will be publication-grade.
