"""Lightweight robust-scaled logistic and shallow tree reliability routers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from .config import EXPERT_NAMES, RouterConfig


def _sklearn_models():
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Method 6 requires scikit-learn. Install it in pe-basline before running."
        ) from error
    return LogisticRegression, HistGradientBoostingClassifier


@dataclass
class RobustFeatureScaler:
    median: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "RobustFeatureScaler":
        finite = np.where(np.isfinite(values), values, np.nan)
        # A diagnostic can be unavailable for every development sample in one
        # LOSO fold.  Such columns are intentionally represented by their
        # missing-value indicators; use a neutral value for the continuous
        # column without emitting NumPy's expected all-NaN warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(finite, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        absolute = np.abs(finite - median)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scale = 1.4826 * np.nanmedian(absolute, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1.0e-9), scale, 1.0)
        self.median = median
        self.scale = scale
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.median is None or self.scale is None:
            raise RuntimeError("Scaler has not been fitted")
        imputed = np.where(np.isfinite(values), values, self.median)
        return (imputed - self.median) / self.scale


@dataclass
class TrainedRouter:
    config: RouterConfig
    scaler: RobustFeatureScaler
    model: Any
    tau_mm: float
    feature_indices: np.ndarray
    feature_names: tuple[str, ...]

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(values[:, self.feature_indices])
        raw = np.asarray(self.model.predict_proba(transformed), dtype=np.float64)
        probabilities = np.zeros((len(values), len(EXPERT_NAMES)), dtype=np.float64)
        for source, class_index in enumerate(self.model.classes_):
            probabilities[:, int(class_index)] = raw[:, source]
        denominator = np.sum(probabilities, axis=1, keepdims=True)
        return probabilities / np.maximum(denominator, 1.0e-12)


def soft_targets(errors: np.ndarray, tau_mm: float) -> np.ndarray:
    shifted = -errors / max(tau_mm, 1.0e-12)
    shifted -= np.max(shifted, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def train_router(
    *,
    features: np.ndarray,
    errors: np.ndarray,
    feature_indices: np.ndarray,
    feature_names: tuple[str, ...],
    config: RouterConfig,
) -> TrainedRouter:
    LogisticRegression, HistGradientBoostingClassifier = _sklearn_models()
    selected = features[:, feature_indices]
    scaler = RobustFeatureScaler().fit(selected)
    transformed = scaler.transform(selected)
    finite_errors = errors[np.isfinite(errors)]
    tau = float(np.median(finite_errors)) if finite_errors.size else 1.0
    if config.soft_target:
        target_probability = soft_targets(errors, tau)
        fit_x = np.repeat(transformed, len(EXPERT_NAMES), axis=0)
        fit_y = np.tile(np.arange(len(EXPERT_NAMES)), len(transformed))
        sample_weight = target_probability.reshape(-1)
    else:
        fit_x = transformed
        fit_y = np.argmin(errors, axis=1)
        sample_weight = None
    if config.model == "logistic":
        model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=config.random_state,
        )
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=config.random_state,
        )
    model.fit(fit_x, fit_y, sample_weight=sample_weight)
    return TrainedRouter(
        config=config,
        scaler=scaler,
        model=model,
        tau_mm=tau,
        feature_indices=feature_indices.copy(),
        feature_names=feature_names,
    )


def smooth_probabilities(probabilities: np.ndarray, alpha: float) -> np.ndarray:
    result = probabilities.copy()
    for index in range(1, len(result)):
        result[index] = alpha * probabilities[index] + (1.0 - alpha) * result[index - 1]
        result[index] /= max(float(np.sum(result[index])), 1.0e-12)
    return result


def mask_invalid_probabilities(
    probabilities: np.ndarray, expert_valid: np.ndarray
) -> np.ndarray:
    masked = probabilities * expert_valid.T
    denominator = np.sum(masked, axis=1, keepdims=True)
    fallback = expert_valid.T.astype(np.float64)
    fallback /= np.maximum(np.sum(fallback, axis=1, keepdims=True), 1.0)
    return np.where(
        denominator > 1.0e-12,
        masked / np.maximum(denominator, 1.0e-12),
        fallback,
    )


def entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1.0e-12, 1.0)
    return -np.sum(safe * np.log(safe), axis=1)


def router_metrics(probabilities: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
    truth = np.argmin(errors, axis=1)
    predicted = np.argmax(probabilities, axis=1)
    confusion = np.zeros((len(EXPERT_NAMES), len(EXPERT_NAMES)), dtype=np.int64)
    for actual, estimate in zip(truth, predicted):
        confusion[int(actual), int(estimate)] += 1
    recall = np.divide(
        np.diag(confusion),
        np.sum(confusion, axis=1),
        out=np.zeros(len(EXPERT_NAMES), dtype=np.float64),
        where=np.sum(confusion, axis=1) > 0,
    )
    one_hot = np.eye(len(EXPERT_NAMES), dtype=np.float64)[truth]
    chosen_error = errors[np.arange(len(errors)), predicted]
    oracle_error = np.min(errors, axis=1)
    values_entropy = entropy(probabilities)
    correlation = (
        float(np.corrcoef(values_entropy, chosen_error)[0, 1])
        if len(values_entropy) > 1
        and np.std(values_entropy) > 0
        and np.std(chosen_error) > 0
        else float("nan")
    )
    return {
        "num_frames": int(len(errors)),
        "accuracy": float(np.mean(predicted == truth)),
        "balanced_accuracy": float(np.mean(recall)),
        "confusion_matrix": confusion.tolist(),
        "log_loss": float(
            -np.mean(np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1)))
        ),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "mean_selected_expert_error_mm": float(np.mean(chosen_error)),
        "mean_oracle_expert_error_mm": float(np.mean(oracle_error)),
        "mean_entropy": float(np.mean(values_entropy)),
        "entropy_selected_error_correlation": correlation,
        "highest_probability_fraction": {
            name: float(np.mean(predicted == index))
            for index, name in enumerate(EXPERT_NAMES)
        },
        "mean_probability": {
            name: float(np.mean(probabilities[:, index]))
            for index, name in enumerate(EXPERT_NAMES)
        },
    }
