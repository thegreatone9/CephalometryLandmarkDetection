#!/usr/bin/env bash
# Train both stages back-to-back.
# Usage: bash train_full_pipeline.sh [--workers N]
set -e

WORKERS=${1:-4}
# Handle --workers flag
if [ "$1" = "--workers" ]; then
    WORKERS=$2
fi

echo "============================================"
echo "  STAGE 1: Coarse Detection (29 landmarks)"
echo "  Workers: $WORKERS"
echo "============================================"

python train.py \
    --encoder resnet34 \
    --epochs 400 \
    --batch-size 12 \
    --img-size 512 \
    --num-workers "$WORKERS"

# Find the Stage 1 checkpoint automatically
STAGE1_CKPT=$(ls -td checkpoints/resnet34-ep400-bs12-img512/best_model.pth 2>/dev/null | head -1)

if [ -z "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found!"
    exit 1
fi

echo ""
echo "============================================"
echo "  STAGE 2: Refinement (crops)"
echo "  Using Stage 1: $STAGE1_CKPT"
echo "  Workers: $WORKERS"
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
