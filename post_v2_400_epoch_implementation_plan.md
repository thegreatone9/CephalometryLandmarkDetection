# Implementation Plan: Better Logging + Next Steps

## Part 1: Enhanced MLflow Logging

Currently we only log `train_loss` and `val_loss` per epoch, plus final evaluation metrics. Here's what's missing and how to add it:

### What to Add

| Metric | Why | Where to add |
|--------|-----|-------------|
| `best_val_loss` | Track the improving floor across epochs (what you showed in the screenshot) | [trainer.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/training/trainer.py#L245-L246) |
| `learning_rate` | See cosine annealing schedule + restart points | [trainer.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/training/trainer.py#L220) |
| `epoch_time_seconds` | Track training speed per epoch | [trainer.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/training/trainer.py#L218) |
| Per-channel val loss | See which landmark channels are learning vs collapsing **during training** | [trainer.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/training/trainer.py#L163-L181) — new method |
| Hardware tags | Know which device ran the training without manual tagging | [train.py](file:///Users/musakhan/Documents/Practice/cephalometry/train.py#L250) |

### Files to Modify

#### [MODIFY] [mlflow_utils.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/training/mlflow_utils.py)
- Add `log_epoch_extended()` function that logs: `best_val_loss`, `learning_rate`, `epoch_time_seconds`
- Add `log_per_channel_val_loss()` function that logs per-landmark channel losses

#### [MODIFY] [trainer.py](file:///Users/musakhan/Documents/Practice/cephalometry/src/training/trainer.py)
- Modify `_validate()` to optionally return per-channel losses (mean loss for each of the 25 heatmap channels)
- In `fit()`, log extended metrics each epoch via `mlflow_utils`
- Log current learning rate from scheduler

#### [MODIFY] [train.py](file:///Users/musakhan/Documents/Practice/cephalometry/train.py)
- Add hardware info to MLflow tags: device type, GPU name, batch size in run tags

> [!NOTE]
> The per-channel val loss is the most valuable addition. It will tell us exactly **when** a channel starts collapsing during training — not just that it failed at the end. This is critical for diagnosing the "attention budget" problem.

---

## Part 2: Next Experiments

Based on the V2 results analysis, here's what to try next, in priority order:

### Experiment 1: Extend to 400 Epochs (Immediate)

**Why:** V2's best val loss was at epoch 195/200 — the model was still improving. The cosine schedule's next long descent phase (epochs 200–340) was never explored.

**What to change:** Just `--epochs 400`. No code changes needed.

**Expected impact:** The 10 failed landmarks may start recovering as the model gets more time. Even if not all recover, we'll have the per-channel loss logs (from Part 1) to see exactly what's happening.

### Experiment 2: Per-Channel Loss Weighting (After Experiment 1 results)

**Why:** The core problem across all runs is that some channels "starve" — the model abandons them. OHLM was the right intuition (weight hard channels more) but the implementation was too aggressive.

**Options to consider (pick one based on Exp 1 results):**
- **Uniform channel weighting:** Average loss per-channel first, then average across channels. This prevents high-pixel-count background from drowning out individual landmarks.
- **Gentle hard-channel boosting:** Like OHLM but with a much smaller amplification (1.5× instead of 3×)

> [!IMPORTANT]
> We should run Experiment 1 with the new logging first. The per-channel loss data will tell us whether the failed channels were never learned or learned-then-forgotten, which determines the right fix.

### Experiment 3: Two-Stage Refinement (If single-stage hits a wall)

**Why:** If after 400 epochs, certain landmarks still fail, the single-stage architecture may genuinely lack capacity for all 25 channels simultaneously.

**Approach:** Train a second lightweight model that takes crops around predicted landmark positions and refines them. This is a well-established approach (Zhong et al., 2019).

**Deferred until:** We have Experiment 1 results with per-channel logging.

---

## Verification Plan

### Automated
- Run training for a few epochs locally, confirm new metrics appear in MLflow
- `mlflow ui` → verify `best_val_loss`, `learning_rate`, per-channel losses are plotted

### Manual
- Review the per-channel loss curves in MLflow after the 400-epoch run completes
