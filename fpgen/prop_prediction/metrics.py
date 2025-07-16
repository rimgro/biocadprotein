# =============================================================================
# prop_prediction/metrics.py
# =============================================================================
# Метрики качества для предсказания свойств белков.
# Включает метрики для классификации и регрессии
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

from typing import Callable, Dict
import numpy as np
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --- Словари с названиями метрик ---

# Регрессия
REGRESSION_METRIC_NAMES: Dict[str, Callable] = {
    'rmse': root_mean_squared_error,
    'mae': mean_absolute_error,
    'r2': r2_score,
    'mae_median': median_absolute_error,
}

# Классификация
CLASSIFICATION_METRIC_NAMES: Dict[str, Callable] = {
    'accuracy': accuracy_score,
    'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
    'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
    'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)
}

# --- Функции для подсчета метрик ---

def get_regression_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:

    '''
    Считает метрики регрессии

    Параметры:
        y_pred, y_true (np.ndarray): метки, которые необходимо предсказать

    Возвращает:
        metrics (dict{str: float}): словарь с метриками
    '''

    return {
        metric_name: metric_fn(y_true, y_pred)
        for metric_name, metric_fn in REGRESSION_METRIC_NAMES.items()
    }

def get_classification_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:

    '''
    Считает метрики классификации

    Параметры:
        y_pred, y_true (np.ndarray): метки, которые необходимо предсказать

    Возвращает:
        metrics (dict{str: float}): словарь с метриками
    '''

    return {
        metric_name: metric_fn(y_true, y_pred)
        for metric_name, metric_fn in CLASSIFICATION_METRIC_NAMES.items()
    }

# --- Bootstrap confidence intervals ---

def bootstrap_metric_ci(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float | Dict[str, float]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, float, float] | Dict[str, tuple[float, float, float]]:
    """Calculate bootstrap confidence intervals for *one or several* metrics.

    The callable ``metric_fn`` can return either a single scalar metric **or**
    a dictionary of metrics (e.g. the output of :pyfunc:`get_regression_metrics`).

    If a scalar is returned, the function behaves exactly as before and returns
    ``(estimate, ci_low, ci_high)``.

    If a ``dict`` is returned, the function returns a *dict* that maps each
    metric name to its corresponding ``(estimate, ci_low, ci_high)`` tuple.
    """

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("y_pred and y_true must be the same length")

    rng = np.random.default_rng(seed=random_state)
    n_samples = y_true.shape[0]

    # Compute point estimate on the full data
    # Please note : get_*_metrics expect (y_pred, y_true) order, so we use the
    # same order here for consistency with those helpers.  Custom metric
    # functions that follow the standard (y_true, y_pred) can be easily wrapped
    # in a lambda if needed.
    point_estimate = metric_fn(y_pred, y_true)

    # --- Case 1: metric_fn returns a dict of metrics ---
    if isinstance(point_estimate, dict):
        # Prepare storage for bootstrap values per metric key
        boot_metrics: Dict[str, np.ndarray] = {
            k: np.empty(n_bootstrap, dtype=float) for k in point_estimate.keys()
        }

        for i in range(n_bootstrap):
            indices = rng.integers(0, n_samples, n_samples)
            boot_estimate_i = metric_fn(y_pred[indices], y_true[indices])
            for k, v in boot_estimate_i.items():
                boot_metrics[k][i] = v

        # Assemble CI for each metric
        results: Dict[str, tuple[float, float, float]] = {}
        for k, arr in boot_metrics.items():
            ci_low = np.percentile(arr, 100 * alpha / 2)
            ci_high = np.percentile(arr, 100 * (1 - alpha / 2))
            results[k] = (point_estimate[k], ci_low, ci_high)

        return results

    # --- Case 2: metric_fn returns a scalar ---
    boot_metrics = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n_samples, n_samples)
        boot_metrics[i] = metric_fn(y_pred[indices], y_true[indices])

    ci_low = np.percentile(boot_metrics, 100 * alpha / 2)
    ci_high = np.percentile(boot_metrics, 100 * (1 - alpha / 2))

    return point_estimate, ci_low, ci_high


def get_regression_metrics_ci(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> Dict[str, tuple[float, float, float]]:
    """Return regression metrics and their confidence intervals.

    The output is a dictionary mapping metric names to a tuple
    ``(point_estimate, ci_low, ci_high)``.
    """
    return {
        name: bootstrap_metric_ci(
            y_pred,
            y_true,
            fn,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            random_state=random_state,
        )
        for name, fn in REGRESSION_METRIC_NAMES.items()
    }


def get_classification_metrics_ci(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> Dict[str, tuple[float, float, float]]:
    """Return classification metrics and their confidence intervals.

    The output is a dictionary mapping metric names to a tuple
    ``(point_estimate, ci_low, ci_high)``.
    """
    return {
        name: bootstrap_metric_ci(
            y_pred,
            y_true,
            fn,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            random_state=random_state,
        )
        for name, fn in CLASSIFICATION_METRIC_NAMES.items()
    }