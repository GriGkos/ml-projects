"""Reusable cross-validation utilities for the Elo regression project."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold


SEED = 42


@dataclass
class CVResult:
    name: str
    mean_rmse: float
    std_rmse: float
    fold_scores: list[float]
    oof_prediction: np.ndarray
    test_prediction: np.ndarray
    params: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("oof_prediction")
        result.pop("test_prediction")
        return result


def load_features(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load cached features and return train/test design matrices without card identifiers."""
    artifacts_dir = project_dir / "artifacts"
    train = pd.read_pickle(artifacts_dir / "train_features.pkl")
    test = pd.read_pickle(artifacts_dir / "test_features.pkl")

    target = train.pop("target").astype("float32")
    test_ids = test[["card_id"]].copy()
    train = train.drop(columns="card_id")
    test = test.drop(columns="card_id")

    all_missing = [column for column in train.columns if train[column].isna().all()]
    if all_missing:
        train = train.drop(columns=all_missing)
        test = test.drop(columns=all_missing)

    # Ratios based on transaction aggregates can be undefined for rare cards.
    # Missing values are a faithful representation and are natively supported
    # by boosting libraries (or later imputed for sklearn estimators).
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    # Tree libraries support missing values. A common float dtype keeps memory
    # reasonable for repeated CV fits and avoids pandas extension dtypes.
    train = train.astype("float32")
    test = test.astype("float32")
    return train, test, target, test_ids


def make_folds(n_splits: int = 5) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def rmse(y_true: pd.Series | np.ndarray, prediction: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, prediction))


def cross_validate_regressor(
    name: str,
    estimator_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    folds: KFold,
    fit_kwargs_factory: Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
) -> CVResult:
    """Fit one model per fold and return OOF/test predictions plus RMSE."""
    oof_prediction = np.zeros(len(X), dtype="float32")
    test_prediction = np.zeros(len(X_test), dtype="float32")
    fold_scores: list[float] = []

    for fold, (train_index, valid_index) in enumerate(folds.split(X), start=1):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model = estimator_factory()
        fit_kwargs = {}
        if fit_kwargs_factory is not None:
            fit_kwargs = fit_kwargs_factory(X_train, y_train, X_valid, y_valid)

        model.fit(X_train, y_train, **fit_kwargs)
        valid_prediction = model.predict(X_valid)
        oof_prediction[valid_index] = valid_prediction
        test_prediction += model.predict(X_test) / folds.n_splits
        score = rmse(y_valid, valid_prediction)
        fold_scores.append(score)
        print(f"{name} | fold {fold}/{folds.n_splits}: RMSE={score:.5f}")

    return CVResult(
        name=name,
        mean_rmse=float(np.mean(fold_scores)),
        std_rmse=float(np.std(fold_scores, ddof=1)),
        fold_scores=fold_scores,
        oof_prediction=oof_prediction,
        test_prediction=test_prediction,
        params=params or {},
    )


def save_result(result: CVResult, output_dir: Path) -> None:
    """Persist predictions separately from the compact experiment summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"oof_{result.name}.npy", result.oof_prediction)
    np.save(output_dir / f"test_{result.name}.npy", result.test_prediction)
    pd.DataFrame([result.summary()]).to_json(
        output_dir / f"summary_{result.name}.json", orient="records", indent=2
    )


def optimize_blend_weights(y: pd.Series, predictions: dict[str, np.ndarray]) -> tuple[dict[str, float], float]:
    """Find non-negative weights summing to one using out-of-fold predictions only."""
    from scipy.optimize import minimize

    names = list(predictions)
    matrix = np.column_stack([predictions[name] for name in names])
    initial = np.full(len(names), 1 / len(names))

    result = minimize(
        lambda weights: rmse(y, matrix @ weights),
        initial,
        method="SLSQP",
        bounds=[(0, 1)] * len(names),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Could not optimize blend: {result.message}")

    weights = {name: float(weight) for name, weight in zip(names, result.x)}
    return weights, rmse(y, matrix @ result.x)
