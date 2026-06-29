#!/usr/bin/env bash
# Train both stages back-to-back, with restart safety.
# If Stage 1 already completed, it will be skipped on rerun.
#
# Usage:
#   bash train_full_pipeline.sh --workers 6
#   bash train_full_pipeline.sh                  # defaults to 4 workers
#   bash train_full_pipeline.sh --stage2-only checkpoints/.../best_model.pth
set -e

WORKERS=4
STAGE2_ONLY=""
# Use a distinct directory name for the 29-landmark two-stage pipeline
STAGE1_CKPT_DIR="checkpoints/stage1-resnet34-ep400-bs12-img512-29L"
STAGE1_CKPT="${STAGE1_CKPT_DIR}/best_model.pth"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --stage2-only)
            STAGE2_ONLY="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

# Override Stage 1 checkpoint if --stage2-only was passed
if [ -n "$STAGE2_ONLY" ]; then
    STAGE1_CKPT="$STAGE2_ONLY"
fi

echo "============================================"
echo "  Two-Stage Pipeline (29 landmarks)"
echo "  Workers: $WORKERS"
echo "============================================"

# Check dependencies
python -c "import mlflow" 2>/dev/null || {
    echo ""
    echo "Installing mlflow..."
    pip install mlflow
}

# ------------------------------------------------------------------
# STAGE 1
# ------------------------------------------------------------------
if [ -f "$STAGE1_CKPT" ] || [ -n "$STAGE2_ONLY" ]; then
    echo ""
    echo "[STAGE 1] SKIPPED — checkpoint already exists: $STAGE1_CKPT"
    echo ""
else
    echo ""
    echo "============================================"
    echo "  STAGE 1: Coarse Detection (29 landmarks)"
    echo "============================================"

    python train.py \
        --encoder resnet34 \
        --epochs 400 \
        --batch-size 12 \
        --img-size 512 \
        --num-workers "$WORKERS" \
        --checkpoint-dir "$STAGE1_CKPT_DIR"

    if [ ! -f "$STAGE1_CKPT" ]; then
        echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CKPT"
        exit 1
    fi
    echo ""
    echo "[STAGE 1] COMPLETE — checkpoint saved: $STAGE1_CKPT"
fi

# ------------------------------------------------------------------
# STAGE 2
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "  STAGE 2: Refinement (crops)"
echo "  Using Stage 1: $STAGE1_CKPT"
echo "============================================"

python train_refiner.py \
    --stage1-checkpoint "$STAGE1_CKPT" \
    --epochs 100 \
    --batch-size 64 \
    --crop-size 96 \
    --jitter 20 \
    --num-workers "$WORKERS"

echo ""
echo "============================================"
echo "  DONE — Both stages complete!"
echo "============================================"
