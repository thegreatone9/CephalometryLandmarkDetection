# Improved vs Baseline: Side-by-Side Comparison

## Headline Numbers

| Metric | Baseline (MSE, 29 lm) | Improved (AWL+OHLM, 25 lm) | Change |
|---|---|---|---|
| Overall MRE | 21.18 mm | 13.97 mm | ↓ 34% better |
| SDR@2mm | 62.7% | 67.5% | ↑ +4.8pp |
| SDR@2.5mm | 66.1% | 71.1% | ↑ +5.0pp |
| SDR@3mm | 68.0% | 73.7% | ↑ +5.7pp |
| SDR@4mm | 70.0% | 77.0% | ↑ +7.0pp |

## Per-Landmark Comparison

### ✅ Previously FAILED → Now FIXED (6 landmarks)

| Landmark | Baseline MRE | Improved MRE | Improvement |
|---|---|---|---|
| **Sella (S)** | 82.25 mm | **0.69 mm** | 🔥 119× better |
| **PNS** | 99.04 mm | **1.56 mm** | 🔥 63× better |
| **Articulare (Ar)** | 81.58 mm | **1.57 mm** | 🔥 52× better |
| **Gonion (Go)** | 67.37 mm | **1.73 mm** | 🔥 39× better |
| **Orbitale (Or)** | 28.66 mm | **2.40 mm** | 🔥 12× better |
| **Condylion (Co)** | 77.22 mm | **2.09 mm** | 🔥 37× better |

> [!NOTE]
> Sella went from 82mm to **0.69mm** — BETTER than its 6-landmark result (0.78mm). AWL+OHLM completely solved the "giving up" problem for skeletal landmarks.

### ⚠️ Still struggling (1 landmark)

| Landmark | Baseline MRE | Improved MRE | Change |
|---|---|---|---|
| **Porion (Po)** | 89.32 mm | **11.47 mm** | 8× better, but still bad |

### ❌ Previously GOOD → Now FAILED (5 dental landmarks)

| Landmark | Baseline MRE | Improved MRE | Change |
|---|---|---|---|
| **Lower Incisor Tip (LIT)** | 1.57 mm | **110.26 mm** | 💀 Collapsed |
| **Lower Incisor Apex (LIA)** | 1.32 mm | **87.82 mm** | 💀 Collapsed |
| **Lower Molar Cusp (LMT)** | 1.61 mm | **52.73 mm** | 💀 Collapsed |
| **Upper Molar Cusp (UMT)** | 2.00 mm | **49.87 mm** | 💀 Collapsed |
| **Upper Incisor Tip (UIT)** | 1.26 mm | **13.09 mm** | 💀 Collapsed |

### ✅ Remained good (13 landmarks)

| Landmark | Baseline | Improved | |
|---|---|---|---|
| Gnathion (Gn) | 0.61 | **0.54** | ↑ |
| Menton (Me) | 0.56 | **0.56** | = |
| Subnasale (Sn) | 0.57 | **0.62** | ≈ |
| Pogonion (Pog) | 0.60 | **0.67** | ≈ |
| Pronasale (Pn) | 0.78 | **0.74** | ↑ |
| Labrale Superius (Ls) | 0.81 | **0.81** | = |
| A-point (A) | 0.89 | **1.00** | ≈ |
| B-point (B) | 0.90 | **0.93** | ≈ |
| Labrale Inferius (Li) | 1.02 | **1.11** | ≈ |
| Soft Tissue Pogonion (Pog') | 1.12 | **1.10** | ≈ |
| Nasion (N) | 0.87 | **1.12** | slightly worse |
| ANS | 1.18 | **1.45** | slightly worse |
| Upper Incisor Apex (UIA) | 1.25 | **3.27** | worse |

---

## Root Cause Analysis: Why Did Dental Landmarks Collapse?

The improved pipeline **fixed skeletal landmarks but broke dental ones**. This is a swap, not a net fix. Why?

### Suspect 1: Adaptive Sigma (σ=3 for dental) — HIGH CONFIDENCE

We set dental landmarks to σ=3 (tighter Gaussian). Combined with:
- **Elastic deformation** (shifts anatomy non-linearly)
- **Affine shifts** (±5% translation)

The narrow σ=3 Gaussian becomes very sensitive to spatial augmentation. A 3px shift that's fine for σ=5 causes the Gaussian peak to fall outside its expected radius at σ=3, creating confusing gradients.

**Fix:** Set all landmarks back to σ=5 (uniform). The dental overlap problem is less damaging than dental landmarks collapsing entirely.

### Suspect 2: OHLM Over-compensation — MODERATE CONFIDENCE

OHLM gives 3× more gradient to hard channels. In early training, skeletal landmarks (upper/posterior) were harder → got 3× gradient → converged fast. Meanwhile, dental landmarks started as "easy" → got 0.5× gradient (minimum cap) → starved of gradient signal → eventually collapsed.

**Fix:** Reduce OHLM clamp range from (0.5, 3.0) to (0.7, 2.0). Less extreme reweighting.

### Suspect 3: Stronger Augmentations — LOWER CONFIDENCE

Elastic deformation at α=30 may distort the dense dental region excessively. Dental landmarks are 5-15px apart; elastic warping can swap their relative positions.

**Fix:** Reduce elastic alpha from 30 to 15, or disable elastic for a test run.

---

## Proposed Next Steps

```
Fix 1: Set ALL landmarks to σ=5 (revert adaptive sigma)
Fix 2: Reduce OHLM clamp from (0.5, 3.0) to (0.7, 2.0)  
Fix 3: Reduce elastic alpha from 30 to 15
```

All three are 1-line changes. The hypothesis: with uniform σ=5 and gentler OHLM, we get the best of both worlds — the skeletal fixes from AWL + the dental accuracy from the baseline.
