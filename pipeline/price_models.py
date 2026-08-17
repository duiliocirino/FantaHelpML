"""FVM -> price model utilities shared by Stage 2A (training) and Stage 2B (inference).

Stage 2A (`02_fvm_to_distribution.ipynb`) trains per-role models and saves them to
``models/fvm_distribution/{SEASON}/model_{mean,std}_{P,D,C,A}.joblib``. Stage 2B
(``03_regressors.ipynb``) loads them to predict ``expprice`` / ``expstd``.

Model formats (see ``docs/phase3_stage2a2b_plan.md``, sections B1/B2):

- mean: sklearn ``Pipeline([StandardScaler, Ridge])`` on features ``[sqrt(FVM), FVM]``
- std:  sklearn ``Pipeline([StandardScaler, Ridge])`` on feature ``[FVM]``
  (Ridge smoothing of binned empirical std)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


def make_features(fvm) -> np.ndarray:
    """Feature matrix for the mean model: columns ``[sqrt(FVM), FVM]``.

    ``sqrt(FVM)`` captures the sub-linear price growth at low FVM; ``FVM``
    captures the acceleration at high FVM (plan B1).
    """
    fvm = np.asarray(fvm, dtype=float)
    return np.column_stack([np.sqrt(fvm), fvm])


def load_price_models(models_dir: str | Path, roles: tuple[str, ...] = ("P", "D", "C", "A")):
    """Load the per-role mean/std models trained by Stage 2A.

    Returns:
        ``(mean_models, std_models)`` — dicts mapping role -> sklearn Pipeline.
    """
    models_dir = Path(models_dir)
    mean_models, std_models = {}, {}
    for role in roles:
        mean_models[role] = joblib.load(models_dir / f"model_mean_{role}.joblib")
        std_models[role] = joblib.load(models_dir / f"model_std_{role}.joblib")
    return mean_models, std_models


def predict_price(mean_models, std_models, fvm, role: str,
                  std_min: float = 1.0, std_max: float = 150.0):
    """Predict ``(price, std)`` for the given FVM values and role.

    The std is clipped to ``[std_min, std_max]`` so every player gets a sane,
    non-zero uncertainty (BE expects positive ints).
    """
    fvm = np.asarray(fvm, dtype=float)
    price = mean_models[role].predict(make_features(fvm))
    std = np.clip(std_models[role].predict(fvm.reshape(-1, 1)), std_min, std_max)
    return price, std
