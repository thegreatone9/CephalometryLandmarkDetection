# Publication Roadmap: Cephalometric Landmark Detection

## Where to Submit — Ranked by Fit for You

| # | Venue | Type | Acceptance Rate | Impact Factor | Deadline | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Nature Scientific Reports** | Journal | ~50% (soundness-based) | 3.8 | Rolling (anytime) | ⭐ **Best fit.** The YOLOv12 paper published here. Same venue = direct comparison. No novelty bar — just technical soundness. |
| 2 | **IEEE ISBI 2027** | Conference | ~45–50% | — | ~Nov 2026 (TBA) | **Strong fit.** CEPHA29 challenge originated here. Short 4-page paper. Good first publication. |
| 3 | **Diagnostics (MDPI)** | Journal | ~moderate | 3.6 | Rolling | **Accessible.** Open access, Q2 journal. Many cephalometry papers here. Faster review. |
| 4 | **BMC Oral Health** | Journal | ~moderate | 2.8 | Rolling | **Clinical angle.** Good if you emphasize clinical utility over architecture novelty. |
| 5 | **MICCAI 2027** | Conference | ~28–30% | — | ~Feb 2027 | **Stretch goal.** Needs strong methodological novelty (not just better numbers). |
| 6 | **Medical Image Analysis** | Journal | ~15–20% | 14.0 | Rolling | **Reach.** Needs major algorithmic contribution. Your work would need a novel architecture component. |

> [!TIP]
> **My recommendation: Nature Scientific Reports.** Rolling submission (submit anytime), soundness-based review (they don't reject for "insufficient novelty"), ~21-day first decision, and the YOLOv12 paper is already there — making your direct comparison maximally impactful. Simultaneously prep a 4-page version for **ISBI 2027** (deadline likely ~Nov 2026).

---

## Step-by-Step Action Plan

### Phase 1: Scale to All 29 Landmarks (Week 1–2)

This is the single highest-impact change. Going from 6→29 landmarks removes the biggest criticism of your current work.

**Code changes needed:**

1. **Update `SELECTED_SYMBOLS`** in `src/data/dataset.py` to include all 29 landmarks
2. **Update `SELECTED_DISPLAY_NAMES`** correspondingly
3. `NUM_LANDMARKS` auto-updates since it's `len(SELECTED_SYMBOLS)`
4. **Update clinical angle computation** in `src/inference/angles.py` — you can now compute more angles (e.g., Frankfort plane using Porion + Orbitale)
5. **Update the Streamlit app** to visualize all 29 landmarks

**Training runs needed:**

| Run | Encoder | Image Size | Epochs | Purpose |
|---|---|---|---|---|
| 1 | ResNet34 | 512 | 100 | Baseline (your proven encoder) |
| 2 | ResNet50 | 512 | 100 | Deeper encoder comparison |
| 3 | ResNet34 | 640 | 100 | Resolution ablation (fix sigma scaling this time) |
| 4 | EfficientNet-b4 | 512 | 100 | Efficient architecture comparison |

> [!IMPORTANT]
> When scaling to 640px, **scale sigma proportionally**: `sigma = 5.0 * (640/512) = 6.25`. This is what caused Sella to fail at 97.89mm in your previous 640px run.

---

### Phase 2: Rigorous Evaluation Framework (Week 2–3)

Reviewers will reject you if you only report overall SDR@2mm. You need:

1. **Per-landmark MRE and SDR** at 2mm, 2.5mm, 3mm, 4mm thresholds (Table format, like the YOLOv12 paper's Table 2)
2. **Bland-Altman plots** for at least 4–6 representative landmarks (Sella, Nasion, Gonion, Orbitale — mix of easy and hard)
3. **Box plots** showing error distribution per landmark
4. **5-fold cross-validation** — this is non-negotiable for a journal paper. A single train/val/test split is not enough.
   - Split the 1000 Aariz images into 5 folds
   - Train 5 models, report mean ± std for all metrics
5. **Statistical significance tests** — paired t-test or Wilcoxon signed-rank test comparing your method vs. a baseline
6. **Clinical angle error propagation** — compute SNA, SNB, ANB from predicted landmarks and compare angle errors to ground truth angles. Report mean absolute angle error ± std.

---

### Phase 3: Comparative Baselines (Week 3–4)

You need to compare against at least 2–3 existing methods:

1. **Your own ablations** (ResNet34 vs ResNet50 vs EfficientNet — you'll have these from Phase 1)
2. **The YOLOv12 paper results** — cite their Table 2 directly for the overlapping landmarks
3. **At least one prior heatmap method** — reimplement or cite results from:
   - The CEPHA29 challenge baselines (same Aariz dataset)
   - Any method from the Aariz dataset paper's benchmark results

> [!NOTE]
> You don't necessarily need to rerun their code. Citing published numbers on the same dataset is standard practice. But using the same test set is critical — if your splits differ, the comparison is invalid. Check if the Aariz dataset has an official train/val/test split.

---

### Phase 4: Write the Paper (Week 4–6)

**Structure** (for Scientific Reports, ~4500 words):

```
Abstract (200 words)
Introduction
  - Clinical motivation
  - Limitations of manual cephalometry
  - Prior work (YOLOv12, other DL approaches)
  - Our contribution (3 bullet points)
Related Work
  - Heatmap regression vs coordinate regression vs bounding box
  - Encoder architectures for medical imaging
  - Pixel spacing calibration (your unique angle)
Methods
  - Dataset (Aariz, 29 landmarks, multi-device)
  - Architecture (U-Net + encoder, heatmap regression)
  - Per-image pixel spacing calibration
  - Training details (optimizer, LR, augmentation, sigma)
  - Evaluation metrics (MRE, SDR, clinical angles)
Results
  - Overall performance table
  - Per-landmark breakdown
  - Encoder comparison
  - Ablation studies (sigma, image size, calibration vs no calibration)
  - Bland-Altman plots
  - Clinical angle accuracy
Discussion
  - Why heatmap regression outperforms bounding box
  - Which landmarks remain challenging and why
  - Clinical implications
  - Limitations
Conclusion
Data Availability Statement
```

---

### Phase 5: Prepare Supplementary Materials (Week 5–6)

1. **Public GitHub repository** (you already have this)
2. **Trained model weights** — upload best checkpoint to Zenodo or Figshare (free DOI)
3. **Supplementary tables** — full 29-landmark results for all encoder variants
4. **Supplementary figures** — prediction visualizations on representative easy/medium/hard cases

---

### Phase 6: Submit (Week 6–7)

1. Create account on Nature's manuscript submission portal
2. Write cover letter highlighting:
   - Direct comparison with published YOLOv12 results
   - Per-image pixel spacing calibration (novel contribution)
   - Full 29 landmarks on Aariz (comprehensive benchmark)
3. Suggest 3–4 reviewers (look at authors of related papers)
4. Submit and wait (~21 days for first decision)

---

## AI Content Policy — What to Watch Out For

### Text

| What | Policy | Safe Practice |
|---|---|---|
| **AI writing full paragraphs** | Must be disclosed in Methods/Acknowledgments | ⚠️ Write the paper yourself. Use AI only for grammar/spelling. |
| **AI grammar/spell checking** | Exempt from disclosure | ✅ Fine, no disclosure needed |
| **AI for literature review** | Must be disclosed | ⚠️ Use AI to *find* papers, but read and cite them yourself. Verify every citation exists — AI hallucinated references are a career-ender. |
| **AI-generated code** | Must be disclosed if it's a core contribution | ⚠️ Your training pipeline used AI assistance. Disclose it: *"Code development was assisted by AI coding tools (Gemini/Claude). All code was reviewed and validated by the authors."* |

### Figures and Diagrams

| What | Policy | Safe Practice |
|---|---|---|
| **AI-generated figures/illustrations** | **BANNED** by Springer Nature | 🚫 Do NOT use DALL-E/Midjourney/etc for any figure |
| **AI-generated graphical abstracts** | **BANNED** | 🚫 Draw them yourself or use proper tools |
| **Matplotlib/Seaborn plots from your data** | Fine (these are data visualizations, not AI art) | ✅ All your training curves, Bland-Altman plots, etc. are fine |
| **Architecture diagrams** | Must be hand-drawn or made with proper tools | ✅ Use draw.io, Lucidchart, PowerPoint, or LaTeX/TikZ |
| **Prediction overlay images** | Fine (these are model outputs on real data) | ✅ Your landmark overlay visualizations are fine |

> [!CAUTION]
> **The #1 risk**: AI-hallucinated citations. If you use AI to help find related work, manually verify every single reference exists by checking the DOI or searching Google Scholar. Fabricated references are grounds for retraction and can end an academic career before it starts.

> [!WARNING]
> **The #2 risk**: Undisclosed AI usage. If reviewers or editors suspect undisclosed AI text generation and you didn't declare it, the paper gets rejected and you may be flagged. Just be transparent — AI-assisted coding and writing is accepted, hiding it is not.

### Recommended Disclosure Statement

Add this to your Methods section:

> *"AI-assisted tools (GitHub Copilot, Claude, Gemini) were used during code development for the training pipeline and web application. All generated code was reviewed, tested, and validated by the authors. AI language models were used for grammar checking and copy-editing of the manuscript text. No AI tools were used to generate figures, data, or scientific conclusions."*

---

## Your Unique Selling Points for the Paper

These are what differentiate your work from existing literature:

1. **Per-image pixel spacing calibration** — Most papers assume uniform 0.1 mm/px. You use the `cephalogram_machine_mappings.csv` to apply per-image correction from 7 different X-ray machines. This is clinically correct and nobody else does it properly.

2. **Full 29 landmarks on Aariz with heatmap regression** — The Aariz paper (2025) introduced the dataset. The YOLOv12 paper (2026) used a different, smaller dataset. You'd be the first comprehensive heatmap regression benchmark on Aariz's full 29.

3. **Multi-encoder ablation** — Systematic comparison of ResNet34/50 and EfficientNet encoders with identical training setup. Most papers test one architecture.

4. **End-to-end clinical pipeline** — Landmark detection → angle computation → clinical interpretation. Most papers stop at landmark detection.

---

## Realistic Timeline

| Week | Phase | Deliverable |
|---|---|---|
| 1–2 | Scale to 29 landmarks + retrain | 4 trained models with full results |
| 2–3 | Evaluation framework | 5-fold CV, Bland-Altman, statistical tests |
| 3–4 | Comparative analysis | Comparison tables, ablation studies |
| 4–6 | Write paper | Full manuscript draft (~4500 words) |
| 5–6 | Supplementary materials | Model weights on Zenodo, supplementary tables |
| 6–7 | Internal review + submit | Final polish, cover letter, submit to Scientific Reports |

**Total: ~7 weeks from today to submission.**

> [!IMPORTANT]
> **Get a co-author.** If you have a faculty advisor or a dentist/orthodontist colleague, bring them on. Clinical validation from a domain expert massively strengthens the paper and is almost expected by reviewers in medical imaging. They can also provide institutional affiliation, which helps with credibility.
