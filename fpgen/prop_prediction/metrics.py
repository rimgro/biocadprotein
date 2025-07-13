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