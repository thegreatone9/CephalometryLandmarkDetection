"""
MLflow integration helpers for experiment tracking.

Wraps common MLflow operations so the training loop can log
hyper-parameters, per-epoch metrics, final evaluation results,
and model checkpoints with minimal boilerplate.
"""

from __future__ import annotations

from typing import Any

import mlflow


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def setup_experiment(
    experiment_name: str = "cephalometric-landmark-detection",
) -> str:
    """Create or retrieve an MLflow experiment by name.

    Parameters
    ----------
    experiment_name : str
        Human-readable experiment name.

    Returns
    -------
    str
        The experiment ID.
    """
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    assert experiment is not None
    return experiment.experiment_id


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_hyperparams(params: dict[str, Any]) -> None:
    """Log a dictionary of hyper-parameters to the active MLflow run.

    Parameters
    ----------
    params : dict
        Key-value pairs (values are auto-converted to strings).
    """
    mlflow.log_params(params)


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    """Log per-epoch training and validation loss.

    Parameters
    ----------
    epoch : int
        Current epoch number (1-indexed).
    train_loss, val_loss : float
        Loss values for the epoch.
    """
    mlflow.log_metrics(
        {"train_loss": train_loss, "val_loss": val_loss},
        step=epoch,
    )


def log_evaluation_metrics(
    mre_per_landmark: dict[str, float],
    mre_overall: float,
    sdr_dict: dict[str, float],
) -> None:
    """Log final evaluation metrics.

    Parameters
    ----------
    mre_per_landmark : dict
        Per-landmark Mean Radial Error (mm).
    mre_overall : float
        Overall MRE across all landmarks.
    sdr_dict : dict
        Successful Detection Rates at various thresholds.
    """
    mlflow.log_metric("mre_overall_mm", mre_overall)
    for name, val in mre_per_landmark.items():
        mlflow.log_metric(f"mre_{name}_mm", val)
    for key, val in sdr_dict.items():
        mlflow.log_metric(key, val)


def log_model_artifact(checkpoint_path: str) -> None:
    """Log a model checkpoint file as an MLflow artifact.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pth`` checkpoint file.
    """
    mlflow.log_artifact(checkpoint_path, artifact_path="model")
