# =============================================================================
# prop_prediction/metrics.py
# =============================================================================
# Метрики качества для предсказания свойств белков.
# Включает метрики для классификации и регрессии
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Авторы: Никита Бакутов, Рим Громов
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

# --- Доверительные интервалы с помощью бутстрепа ---

def bootstrap_metric_ci(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float | Dict[str, float]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, float, float] | Dict[str, tuple[float, float, float]]:
    
    '''
    Вычисляет доверительные интервалы с помощью бутстрепа для одной или нескольких метрик.

    Функция metric_fn может возвращать как скалярное значение метрики, так и 
    словарь метрик (например, результат работы get_regression_metrics).

    Если возвращается скаляр, функция возвращает кортеж (оценка, нижняя_граница_ci, верхняя_граница_ci).

    Если возвращается словарь, функция возвращает словарь, где каждой метрике
    соответствует кортеж (оценка, нижняя_граница_ci, верхняя_граница_ci).

    Параметры:
        y_pred, y_true (np.ndarray): предсказанные и истинные метки
        metric_fn (Callable): функция для вычисления метрик
        n_bootstrap (int): количество бутстреп-выборок
        alpha (float): уровень значимости
        random_state (int | None): seed для генератора случайных чисел

    Возвращает:
        Кортеж или словарь с оценкой метрики и границами доверительного интервала
    '''

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("y_pred и y_true должны иметь одинаковую длину")

    rng = np.random.default_rng(seed=random_state)
    n_samples = y_true.shape[0]

    # Вычисление точечной оценки на полных данных
    point_estimate = metric_fn(y_pred, y_true)

    # --- Случай 1: metric_fn возвращает словарь метрик ---
    if isinstance(point_estimate, dict):
        # Подготовка массива для хранения бутстреп-значений
        boot_metrics: Dict[str, np.ndarray] = {
            k: np.empty(n_bootstrap, dtype=float) for k in point_estimate.keys()
        }

        for i in range(n_bootstrap):
            indices = rng.integers(0, n_samples, n_samples)
            boot_estimate_i = metric_fn(y_pred[indices], y_true[indices])
            for k, v in boot_estimate_i.items():
                boot_metrics[k][i] = v

        # Формирование доверительных интервалов для каждой метрики
        results: Dict[str, tuple[float, float, float]] = {}
        for k, arr in boot_metrics.items():
            ci_low = np.percentile(arr, 100 * alpha / 2)
            ci_high = np.percentile(arr, 100 * (1 - alpha / 2))
            results[k] = (point_estimate[k], ci_low, ci_high)

        return results

    # --- Случай 2: metric_fn возвращает скаляр ---
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
    
    '''
    Возвращает метрики регрессии с их доверительными интервалами.

    Возвращает словарь, где каждой метрике соответствует кортеж
    (точечная_оценка, нижняя_граница_ci, верхняя_граница_ci).

    Параметры:
        y_pred, y_true (np.ndarray): предсказанные и истинные метки
        n_bootstrap (int): количество бутстреп-выборок
        alpha (float): уровень значимости
        random_state (int | None): seed для генератора случайных чисел

    Возвращает:
        Словарь с метриками и их доверительными интервалами
    '''

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
    
    '''
    Возвращает метрики классификации с их доверительными интервалами.

    Возвращает словарь, где каждой метрике соответствует кортеж
    (точечная_оценка, нижняя_граница_ci, верхняя_граница_ci).

    Параметры:
        y_pred, y_true (np.ndarray): предсказанные и истинные метки
        n_bootstrap (int): количество бутстреп-выборок
        alpha (float): уровень значимости
        random_state (int | None): seed для генератора случайных чисел

    Возвращает:
        Словарь с метриками и их доверительными интервалами
    '''

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