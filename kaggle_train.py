"""
Kaggle Training Notebook — Cephalometric Landmark Detection
============================================================
Run this as a Kaggle notebook with GPU (T4) enabled.

Prerequisites:
  - Dataset "aariz-cephalometric" uploaded to Kaggle (see README below)
  - GPU T4 enabled in notebook settings
"""

# ============================================================
# 1. Install dependencies
# ============================================================
!pip install -q segmentation_models_pytorch albumentations mlflow

# ============================================================
# 2. Clone the repo and checkout the right branch
# ============================================================
!git clone https://github.com/thegreatone9/CephalometryLandmarkDetection.git /kaggle/working/ceph
%cd /kaggle/working/ceph
!git checkout improved-training-v2

# ============================================================
# 3. Symlink the data
# ============================================================
# Kaggle datasets are mounted at /kaggle/input/<dataset-name>/
# Adjust DATASET_NAME to match what you named your upload
import os

DATASET_NAME = "aariz-cephalometric"  # <-- change if you named it differently
KAGGLE_DATA = f"/kaggle/input/{DATASET_NAME}"

# The data directory structure should be:
#   /kaggle/input/aariz-cephalometric/
#   ├── train/Cephalograms/ + Annotations/
#   ├── valid/Cephalograms/ + Annotations/
#   ├── test/Cephalograms/ + Annotations/
#   └── cephalogram_machine_mappings.csv

# Create symlink so train.py finds data/ in the expected location
os.symlink(KAGGLE_DATA, "/kaggle/working/ceph/data")

# Verify
print("Data structure:")
for split in ["train", "valid", "test"]:
    imgs = len(os.listdir(f"/kaggle/working/ceph/data/{split}/Cephalograms"))
    print(f"  {split}: {imgs} images")
print(f"  CSV: {os.path.exists('/kaggle/working/ceph/data/cephalogram_machine_mappings.csv')}")

# ============================================================
# 4. Run training
# ============================================================
!python train.py --encoder resnet34 --epochs 200 --batch-size 8 --img-size 512

# ============================================================
# 5. Save outputs (Kaggle auto-saves /kaggle/working/)
# ============================================================
# The checkpoint and MLflow DB are already in /kaggle/working/ceph/
print("\n=== Training complete! ===")
print("Checkpoint: checkpoints/resnet34-ep200-bs8-img512/best_model.pth")
print("MLflow DB:  mlflow.db")
