# V2-400 Training Results — Objective Analysis

## TL;DR

400 epochs is a **clear improvement** over 200 epochs, but the "rotating collapse" problem persists. More landmarks recovered than collapsed, but the model still can't hold all 25 channels simultaneously.

---

## Overall Metrics Across All Runs

| Run | Loss | Epochs | Overall MRE | Good (<3mm) | Moderate (3–10mm) | Failed (>10mm) | SDR@2mm | SDR@4mm |
|-----|------|--------|-------------|-------------|-------------------|----------------|---------|---------|
| Baseline | MSE | 200 | 21.18mm | 21/29 | 0 | 8 | — | — |
| V1 | AWL+OHLM | 200 | 13.97mm | 18/25 | 1 | 6 | — | — |
| V2-200 | AWL | 200 | 13.54mm | 13/25 | 2 | 10 | 69.1% | 76.8% |
| **V2-400** | **AWL** | **400** | **9.15mm** | **15/25** | **4** | **6** | **74.8%** | **85.0%** |

> [!NOTE]
> V2-400's overall MRE (9.15mm) is the best across all runs. SDR@4mm jumped from 76.8% to 85.0%.

---

## Per-Landmark Comparison (sorted by V2-400 performance)

| Landmark | Baseline | V1 | V2-200 | **V2-400** | Trend |
|----------|---------|-------|---------|------------|-------|
| Me | 0.56 ✅ | 0.56 ✅ | 0.47 ✅ | **0.53** ✅ | Stable |
| Gn | 0.61 ✅ | 0.54 ✅ | 0.51 ✅ | **0.53** ✅ | Stable |
| Pog | 0.60 ✅ | 0.67 ✅ | 0.61 ✅ | **0.73** ✅ | Stable |
| Sn | 0.57 ✅ | 0.62 ✅ | 0.74 ✅ | **0.76** ✅ | Stable |
| LIT | 1.57 ✅ | 110.26 ❌ | 18.97 ❌ | **0.77** ✅ | ★ Recovered |
| Pn | 0.78 ✅ | 0.74 ✅ | 20.91 ❌ | **0.82** ✅ | ★ Recovered |
| B | 0.90 ✅ | 0.93 ✅ | 0.96 ✅ | **0.91** ✅ | Stable |
| Pog' | 1.12 ✅ | 1.10 ✅ | 1.13 ✅ | **1.16** ✅ | Stable |
| UIT | 1.26 ✅ | 13.09 ❌ | 0.64 ✅ | **1.26** ✅ | Stable |
| N | 0.87 ✅ | 1.12 ✅ | 44.54 ❌ | **1.31** ✅ | ★ Recovered |
| A | 0.89 ✅ | 1.00 ✅ | 0.90 ✅ | **1.33** ✅ | Stable |
| ANS | 1.18 ✅ | 1.45 ✅ | 1.45 ✅ | **1.45** ✅ | Stable |
| UIA | 1.25 ✅ | 3.27 ⚠️ | 1.24 ✅ | **1.50** ✅ | Stable |
| S | 82.25 ❌ | 0.69 ✅ | 6.06 ⚠️ | **1.77** ✅ | ★ Recovered |
| Or | 28.66 ❌ | 2.40 ✅ | 38.42 ❌ | **2.80** ✅ | ★ Recovered |
| PNS | 99.04 ❌ | 1.56 ✅ | 11.21 ❌ | **3.39** ⚠️ | Improving |
| Co | 77.22 ❌ | 2.09 ✅ | 2.14 ✅ | **5.87** ⚠️ | Regressing |
| Po | 89.32 ❌ | 11.47 ❌ | 60.90 ❌ | **6.73** ⚠️ | ★ Best ever |
| Ls | 0.81 ✅ | 0.81 ✅ | 0.74 ✅ | **9.72** ⚠️ | ⚠ Near-collapse |
| Ar | 81.58 ❌ | 1.57 ✅ | 3.75 ⚠️ | **12.89** ❌ | Regressing |
| LMT | 1.61 ✅ | 52.73 ❌ | 20.59 ❌ | **15.17** ❌ | Improving slowly |
| UMT | 2.00 ✅ | 49.87 ❌ | 20.39 ❌ | **15.38** ❌ | Improving slowly |
| Li | 1.02 ✅ | 1.11 ✅ | 17.64 ❌ | **24.18** ❌ | Getting worse |
| LIA | 1.32 ✅ | 87.82 ❌ | 1.40 ✅ | **48.51** ❌ | ⚠ Collapsed |
| Go | 67.37 ❌ | 1.73 ✅ | 62.11 ❌ | **69.25** ❌ | Stubbornly bad |

---

## What Changed from V2-200 → V2-400

### ✅ Recovered (5 landmarks)
| Landmark | V2-200 | V2-400 | Improvement |
|----------|--------|--------|-------------|
| N (Nasion) | 44.54 ❌ | 1.31 ✅ | −43.2mm |
| Or (Orbitale) | 38.42 ❌ | 2.80 ✅ | −35.6mm |
| Pn (Pronasale) | 20.91 ❌ | 0.82 ✅ | −20.1mm |
| LIT (Lower Incisor Tip) | 18.97 ❌ | 0.77 ✅ | −18.2mm |
| Po (Porion) | 60.90 ❌ | 6.73 ⚠️ | −54.2mm |

### ❌ Collapsed (3 landmarks)
| Landmark | V2-200 | V2-400 | Regression |
|----------|--------|--------|-----------|
| LIA (Lower Incisor Apex) | 1.40 ✅ | 48.51 ❌ | +47.1mm |
| Ls (Labrale Superius) | 0.74 ✅ | 9.72 ⚠️ | +9.0mm |
| Co (Condylion) | 2.14 ✅ | 5.87 ⚠️ | +3.7mm |

---

## Per-Channel Loss Analysis

The per-channel MSE validation loss data (logged every 10 epochs) reveals something surprising:

> [!IMPORTANT]
> **All channels converge to similar loss values regardless of whether the landmark is accurately detected or not.** Go (69mm MRE, loss 0.00043) and Gn (0.53mm MRE, loss 0.00008) are only ~5× apart in loss, but ~130× apart in MRE. This means the model learns to predict near-blank heatmaps that satisfy the per-pixel loss function without actually detecting the landmark.

This confirms the failure mode: when a channel "collapses," the model doesn't produce a wrong prediction — it produces **no prediction** (a near-zero heatmap). The loss function can't distinguish between "correctly predicted nothing because there's no landmark" and "gave up and predicted blank."

---

## Landmark Stability Classification

Based on behaviour across all 4 runs:

| Category | Landmarks | Count |
|----------|-----------|-------|
| **Always reliable** (<3mm in all runs) | Me, Gn, Pog, B, ANS, Pog', Sn | 7 |
| **Usually reliable** (<3mm in 3/4 runs) | A, Pn, UIT, UIA, Ls | 5 |
| **Unstable** (flips between good and bad) | N, Or, S, LIT, LIA, Li, Co, PNS | 8 |
| **Usually bad** (>10mm in 3/4 runs) | Go, Po, Ar, LMT, UMT | 5 |

---

## Objective Verdict

**The good:**
- 400 epochs is definitively better than 200. Overall MRE dropped 32% (13.54 → 9.15mm).
- 5 previously-failed landmarks recovered, including dramatic saves (N: 44→1.3mm, Po: 61→6.7mm).
- SDR@2mm: 74.8% is approaching competitive territory.
- The user's intuition about "it was still improving" was correct and paid off.

**The bad:**
- 3 landmarks that were good at 200 epochs collapsed by epoch 400 (LIA: 1.4→48.5mm is the worst).
- 6 landmarks still fail (>10mm), with Go (69mm) stubbornly refusing to learn.
- The "rotating collapse" pattern means we cannot guarantee which landmarks will work in any given training run.

**The fundamental issue:**
This is not a training duration or loss function problem — it's an **architectural capacity problem**. A single ResNet34 U-Net with shared features cannot simultaneously produce accurate heatmaps for all 25 landmarks. The model has ~24M parameters spread across 25 output channels, and it consistently "drops" 5-10 of them regardless of training configuration.

**What this means for the paper:**
The current single-stage approach will not achieve publication-quality results across all landmarks. The per-landmark variability is too high and the failure modes are too unpredictable. A different architectural approach is needed.
