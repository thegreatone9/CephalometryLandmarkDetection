# Reviewer Criteria Scorecard: Your Project vs. What Gets Accepted

Based on the CLAIM checklist (Checklist for AI in Medical Imaging), Scientific Reports editorial criteria, and common rejection reasons from medical imaging peer review.

---

## The 10 Criteria Reviewers Use

### 1. Technical Soundness of Method

**What they look for:** Is the architecture well-justified? Is heatmap regression appropriate for this task? Are training details complete (optimizer, LR schedule, augmentation, loss function)?

| Item | Your Status | Notes |
|---|---|---|
| Architecture justified | ✅ Done | U-Net + pretrained encoder is standard and well-cited for this task |
| Heatmap regression rationale | ✅ Done | Clearly superior to bounding-box; YOLOv12 paper itself concedes this |
| Training hyperparameters documented | ✅ Done | Encoder/decoder LR, sigma, batch size, image size all in `train.py` |
| Loss function specified | ✅ Done | MSE loss on Gaussian heatmaps |
| Data augmentation described | ✅ Done | Albumentations pipeline in `preprocessing.py` |
| Transfer learning justified | ✅ Done | ImageNet pretrained encoder |

**Verdict: ✅ Strong. No gaps.**

---

### 2. Evaluation Rigor

**What they look for:** Are the right metrics used? Are results reported at multiple thresholds? Is there per-landmark breakdown?

| Item | Your Status | Notes |
|---|---|---|
| MRE (mm) reported | ✅ Done | 0.78mm overall |
| SDR@2mm reported | ✅ Done | 94.9% |
| SDR at multiple thresholds (2, 2.5, 3, 4mm) | ❌ Missing | **Reviewers expect at least 3 thresholds** |
| Per-landmark MRE breakdown | ✅ Done | In MLflow logs and report |
| Per-landmark SDR breakdown | ❌ Missing | **Must add — this is Table 2 in every accepted paper** |
| Error distribution visualization | ❌ Missing | Box plots or violin plots per landmark |
| Bland-Altman plots | ❌ Missing | **Required for clinical credibility — every 2026 paper has these** |

**Verdict: ⚠️ Partial. The overall numbers are there but the detailed breakdowns that reviewers expect are missing.**

> [!IMPORTANT]
> **Priority fix:** Add SDR at 2mm/2.5mm/3mm/4mm per landmark, and generate Bland-Altman plots for at least 6 representative landmarks. This is the #1 thing that separates a course project from a publishable paper.

---

### 3. Statistical Validation

**What they look for:** Are results statistically robust? Cross-validation? Confidence intervals? Significance tests?

| Item | Your Status | Notes |
|---|---|---|
| Cross-validation (k-fold) | ❌ Missing | **Single train/val/test split only — #1 rejection risk** |
| Confidence intervals on MRE/SDR | ❌ Missing | Must report mean ± std across folds |
| Statistical significance test | ❌ Missing | Paired t-test or Wilcoxon between encoders |
| Multiple random seeds | ❌ Missing | At minimum, train 3 times and report variance |

**Verdict: ❌ Weak. This is probably your biggest gap right now.**

> [!CAUTION]
> **This is the #1 reason medical imaging papers get rejected.** A single train/test split with a single number is not publishable evidence. You need either 5-fold cross-validation or at minimum 3 independent runs with different seeds to report mean ± std.

---

### 4. Dataset Quality & Documentation

**What they look for:** Is the dataset well-described? Are annotation procedures documented? Is there potential bias?

| Item | Your Status | Notes |
|---|---|---|
| Dataset source cited | ✅ Done | Aariz (Khalid et al., Scientific Data, 2025) |
| Dataset size and splits documented | ✅ Done | 1000 images, train/val/test |
| Annotation procedure described | ✅ Done | Junior + Senior orthodontist annotations averaged |
| Multi-device diversity acknowledged | ✅ Done | 7 X-ray machines with varying pixel spacings |
| Pixel spacing calibration | ✅ Done | **Unique strength — per-image mm/px from CSV** |
| Demographic information | ⚠️ Partial | Aariz paper describes demographics; cite it, add a sentence in Methods |
| Inclusion/exclusion criteria | ⚠️ Partial | Mention you used all 1000 images without exclusion |

**Verdict: ✅ Strong. The Aariz dataset is well-documented and your pixel spacing work is a genuine differentiator.**

---

### 5. Clinical Relevance

**What they look for:** Does this actually matter to clinicians? How do landmark errors affect real clinical decisions?

| Item | Your Status | Notes |
|---|---|---|
| Clinical motivation stated | ✅ Done | README and final report both explain orthodontic context |
| Clinical angle computation | ✅ Done | SNA, SNB, ANB computed in `angles.py` |
| Angle error analysis (predicted vs ground truth) | ❌ Missing | **This is your secret weapon — compute it and add it** |
| Clinical interpretation of results | ✅ Done | Streamlit app has plain-English interpretation |
| Comparison to manual tracing variability | ❌ Missing | Cite literature on inter-observer error (~1–2mm) and compare |

**Verdict: ⚠️ Partial. The angle computation exists in your code but you haven't run it as a formal evaluation metric.**

> [!TIP]
> **Easy win:** Compute SNA, SNB, ANB angles from predicted landmarks and from ground truth landmarks, then report the mean absolute angle error (°). This is something the YOLOv12 paper didn't do at all, and the YOLOv11 preprint only did for 4 landmarks. If your angle errors are within ±2°, that's clinically acceptable and very publishable.

---

### 6. Reproducibility & Open Science

**What they look for:** Can someone else reproduce your results? Code available? Models available?

| Item | Your Status | Notes |
|---|---|---|
| Code publicly available | ✅ Done | GitHub repo is public |
| Dataset publicly available | ✅ Done | Aariz is on Figshare |
| Trained model weights shared | ❌ Missing | **Upload best checkpoint to Zenodo/Figshare with a DOI** |
| Environment/dependencies documented | ✅ Done | `requirements.txt` and Dockerfile |
| Docker containerization | ✅ Done | **Bonus — most papers don't do this** |
| Random seed fixed and reported | ⚠️ Partial | Check if seed is set in training code |

**Verdict: ✅ Strong. Better than most published papers. Just need to upload model weights.**

---

### 7. Comparison to Prior Work

**What they look for:** Fair comparison to existing methods? Using the same dataset and metrics?

| Item | Your Status | Notes |
|---|---|---|
| Comparison to at least 2 baselines | ❌ Missing | **Need to cite at least YOLOv12 + one heatmap method** |
| Same dataset used for comparison | ⚠️ Tricky | YOLOv12 used a different dataset; cite their numbers but acknowledge this |
| Ablation study (encoder variants) | ✅ Done | ResNet34 vs EfficientNet-b0 (add ResNet50) |
| Fair comparison conditions | ⚠️ Partial | Your ablations use identical setup — good. External comparisons need caveats. |

**Verdict: ⚠️ Partial. You have internal ablations but need to position against external work more formally.**

> [!NOTE]
> Since the YOLOv12 paper used a different dataset, you can't make a direct apples-to-apples comparison. But you **can**:
> 1. Compare on overlapping landmarks (Sella, Nasion, Menton, Pogonion) — both papers detect these
> 2. Cite CephRes-MHNet's numbers on Aariz (1.23mm MRE, 85.5% SDR@2mm) since they used the same dataset, even though the paper was withdrawn
> 3. Run your own ablation as the primary comparison (ResNet34 vs ResNet50 vs EfficientNet)

---

### 8. Robustness & Generalizability

**What they look for:** Does the model work across different conditions? Different scanners? Different patient populations?

| Item | Your Status | Notes |
|---|---|---|
| Multi-device evaluation | ⚠️ Partial | Aariz has 7 devices, but you don't report per-device results |
| Failure case analysis | ❌ Missing | **Show 3–4 examples where the model fails and explain why** |
| Sensitivity to image quality | ❌ Missing | What happens with overexposed/underexposed X-rays? |
| Per-device SDR breakdown | ❌ Missing | **Easy to compute from the machine mappings CSV — strong contribution** |

**Verdict: ⚠️ Partial. The multi-device data is there but you haven't exploited it.**

> [!TIP]
> **Easy and novel contribution:** Since you have per-image device information from `cephalogram_machine_mappings.csv`, compute SDR@2mm broken down by X-ray machine. This directly addresses the "domain shift" concern that reviewers care about, and **no published paper has done this on Aariz.**

---

### 9. Reporting Standards (CLAIM Checklist)

**What they look for:** Did you follow the CLAIM (Checklist for AI in Medical Imaging) reporting standard?

| CLAIM Item | Your Status |
|---|---|
| Title identifies AI/DL method | ✅ |
| Abstract reports key performance metrics | ✅ |
| Study design described | ✅ |
| Data sources described | ✅ |
| Eligibility criteria stated | ⚠️ Add a sentence |
| Data preprocessing described | ✅ |
| Data partitions described | ✅ |
| Model architecture described | ✅ |
| Training procedure described | ✅ |
| Evaluation metrics defined | ✅ |
| Statistical measures of performance reported | ❌ Need confidence intervals |
| Failure analysis conducted | ❌ Missing |
| CLAIM checklist submitted with manuscript | ❌ Need to fill out and submit |

**Verdict: ⚠️ Mostly compliant. Fill out the official CLAIM Word template before submission.**

---

### 10. Writing Quality & Structure

**What they look for:** Clear, concise, follows journal format, proper citations, no AI-generated fluff.

| Item | Your Status | Notes |
|---|---|---|
| Follows journal structure | ⚠️ Not yet | Need to rewrite for Sci Reports format (~4500 words) |
| Proper citation format | ⚠️ Not yet | Need BibTeX/reference manager |
| Figures are publication quality | ⚠️ Partial | Your Matplotlib plots need polish (300 DPI, proper labels, consistent style) |
| Tables formatted correctly | ⚠️ Not yet | Need LaTeX or journal template formatting |
| No AI-generated prose | ⚠️ Watch out | Write the paper yourself; only use AI for grammar |

**Verdict: ⚠️ This is normal — you haven't written the paper yet. Just be aware of the standards.**

---

## Prioritized Gap Closure List

Ranked by **impact on acceptance** (what reviewers will reject you for first):

| Priority | Gap | Effort | Impact on Acceptance |
|---|---|---|---|
| 🔴 1 | **5-fold cross-validation** | ~5× training time (heavy) | Fatal if missing — instant reject at any serious venue |
| 🔴 2 | **Per-landmark SDR at 2/2.5/3/4mm thresholds** | Light (code change in `evaluation.py`) | Expected in every paper — reviewers check for this immediately |
| 🔴 3 | **Bland-Altman plots** (6+ landmarks) | Light (matplotlib script) | Clinical credibility signal — all 2026 papers have these |
| 🟡 4 | **Scale to 29 landmarks** | Medium (code change + retrain) | Moves you from "student project" to "benchmark paper" |
| 🟡 5 | **Clinical angle error analysis** | Light (you already have the code) | Differentiator — YOLOv12 paper didn't do this |
| 🟡 6 | **Failure case analysis** with examples | Light (pick 3-4 bad predictions, explain) | Shows maturity and self-awareness |
| 🟡 7 | **Per-device SDR breakdown** | Light (group by machine from CSV) | Novel contribution — nobody has done this |
| 🟢 8 | **Add ResNet50 encoder run** | Medium (1 more training run) | Strengthens ablation study |
| 🟢 9 | **Upload model weights to Zenodo** | Trivial | Reproducibility checkbox |
| 🟢 10 | **Fill out CLAIM checklist** | Light (30-min form) | Some journals require it with submission |

---

## How to Excel (Go Beyond Baseline Acceptance)

These aren't required for acceptance, but would make your paper stand out:

1. **Per-device performance analysis** — Group test results by X-ray machine using your CSV metadata. Show that your model generalizes across devices. **Nobody has published this on Aariz.**

2. **Uncertainty estimation** — Add Monte Carlo dropout (run inference 10× with dropout enabled, report std of predictions). Reviewers in 2026 specifically ask for this. ~2 hours of code changes.

3. **Heatmap confidence visualization** — Show the predicted heatmap overlaid on the X-ray for a few examples. Reviewers love seeing what the model "looks at." You already have this data — just visualize it.

4. **Pixel spacing ablation** — Run evaluation twice: once with per-image pixel spacing, once with a flat 0.1 mm/px assumption. Show the difference. This proves your calibration actually matters.

5. **Inference speed benchmark** — Report predictions/second on CPU and GPU. If your model does 30+ FPS, mention it — real-time capability is a publishable selling point.
