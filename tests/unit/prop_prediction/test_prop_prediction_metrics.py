import numpy as np
import pytest
from fpgen.prop_prediction.metrics import (
    get_regression_metrics,
    get_classification_metrics,
    bootstrap_metric_ci,
    get_regression_metrics_ci,
    get_classification_metrics_ci,
    REGRESSION_METRIC_NAMES,
    CLASSIFICATION_METRIC_NAMES
)

# --- Тесты для регрессии ---

def test_get_regression_metrics_perfect():
    """Тест для идеального предсказания в регрессии"""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = get_regression_metrics(y_pred, y_true)
    
    assert np.isclose(metrics['rmse'], 0.0)
    assert np.isclose(metrics['mae'], 0.0)
    assert np.isclose(metrics['r2'], 1.0)
    assert np.isclose(metrics['mae_median'], 0.0)

def test_get_regression_metrics_imperfect():
    """Тест для неидеального предсказания в регрессии"""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 1, 1, 2])
    y_pred = np.array([1, 3, 1, 1, 1, 2, 1, 1, 3, 2])
    metrics = get_regression_metrics(y_pred, y_true)

    assert np.isclose(metrics['rmse'], 1.0488088481701516)
    assert np.isclose(metrics['mae'], 0.7)
    assert np.isclose(metrics['r2'], -0.80327868852459)
    assert np.isclose(metrics['mae_median'], 0.5)

def test_get_regression_metrics_shape_mismatch():
    """Тест на проверку формы входных данных"""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])
    with pytest.raises(ValueError):
        get_regression_metrics(y_pred, y_true)

# --- Тесты для классификации ---

def test_get_classification_metrics_perfect():
    """Тест для идеального предсказания в классификации"""
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 2, 1])
    metrics = get_classification_metrics(y_pred, y_true)

    assert np.isclose(metrics['accuracy'], 1.0)
    assert np.isclose(metrics['precision'], 1.0)
    assert np.isclose(metrics['recall'], 1.0)
    assert np.isclose(metrics['f1'], 1.0)

def test_get_classification_metrics_imperfect():
    """Тест для неидеального предсказания в классификации"""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 1, 1, 2])
    y_pred = np.array([1, 3, 1, 1, 1, 2, 1, 1, 3, 2])
    metrics = get_classification_metrics(y_pred, y_true)

    assert np.isclose(metrics['accuracy'], 0.5)
    assert np.isclose(metrics['precision'], 0.4833333333333333)
    assert np.isclose(metrics['recall'], 0.5)
    assert np.isclose(metrics['f1'], 0.4836363636363637)

def test_get_classification_metrics_shape_mismatch():
    """Тест на проверку формы входных данных"""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])
    with pytest.raises(ValueError):
        get_classification_metrics(y_pred, y_true)

# --- Тесты для доверительных интервалов ---

def test_bootstrap_metric_ci_scalar():
    """Тест для скалярной метрики с бутстрепом"""
    np.random.seed(42)
    y_true = np.random.rand(100)
    y_pred = y_true + np.random.normal(0, 0.1, 100)
    
    def mse(y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()
    
    result = bootstrap_metric_ci(y_pred, y_true, mse, n_bootstrap=100)
    assert len(result) == 3  # (estimate, ci_low, ci_high)
    assert result[0] <= result[2]  # estimate <= ci_high
    assert result[0] >= result[1]  # estimate >= ci_low

def test_bootstrap_metric_ci_dict():
    """Тест для словаря метрик с бутстрепом"""
    np.random.seed(42)
    y_true = np.random.rand(100)
    y_pred = y_true + np.random.normal(0, 0.1, 100)
    
    def multi_metrics(y_true, y_pred):
        return {
            'mse': ((y_true - y_pred) ** 2).mean(),
            'mae': np.abs(y_true - y_pred).mean()
        }
    
    results = bootstrap_metric_ci(y_pred, y_true, multi_metrics, n_bootstrap=100)
    
    assert isinstance(results, dict)
    assert 'mse' in results
    assert 'mae' in results
    assert len(results['mse']) == 3
    assert results['mse'][0] <= results['mse'][2]
    assert results['mse'][0] >= results['mse'][1]

def test_bootstrap_metric_ci_shape_mismatch():
    """Тест на проверку формы входных данных"""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])
    with pytest.raises(ValueError):
        bootstrap_metric_ci(y_pred, y_true, lambda y_true, y_pred: 0)

def test_get_regression_metrics_ci():
    """Тест для доверительных интервалов метрик регрессии"""
    np.random.seed(42)
    y_true = np.random.rand(100)
    y_pred = y_true + np.random.normal(0, 0.1, 100)
    
    results = get_regression_metrics_ci(y_pred, y_true, n_bootstrap=100)
    
    assert isinstance(results, dict)
    assert set(results.keys()) == set(REGRESSION_METRIC_NAMES.keys())
    for metric_name, (estimate, ci_low, ci_high) in results.items():
        assert estimate <= ci_high
        assert estimate >= ci_low

def test_get_classification_metrics_ci():
    """Тест для доверительных интервалов метрик классификации"""
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    
    results = get_classification_metrics_ci(y_pred, y_true, n_bootstrap=100)
    
    assert isinstance(results, dict)
    assert set(results.keys()) == set(CLASSIFICATION_METRIC_NAMES.keys())
    for metric_name, (estimate, ci_low, ci_high) in results.items():
        assert estimate <= ci_high
        assert estimate >= ci_low

def test_bootstrap_with_small_sample():
    """Тест с маленькой выборкой"""
    y_true = np.array([1, 2])
    y_pred = np.array([1, 2])
    
    # Должно работать даже с маленькой выборкой
    result = bootstrap_metric_ci(y_pred, y_true, lambda y_true, y_pred: 1, n_bootstrap=10)
    assert result == (1, 1, 1)  # Все бутстреп выборки будут одинаковы

def test_bootstrap_with_zero_variance():
    """Тест с нулевой дисперсией"""
    y_true = np.ones(10)
    y_pred = np.ones(10)
    
    results = get_regression_metrics_ci(y_pred, y_true, n_bootstrap=100)
    
    for metric_name, (estimate, ci_low, ci_high) in results.items():
        if metric_name == 'r2':
            # R2 может быть NaN в этом случае
            continue
        assert np.isclose(estimate, ci_low)
        assert np.isclose(estimate, ci_high)

# --- Тесты на граничные случаи ---

def test_empty_input():
    """Тест с пустыми входными данными"""
    y_true = np.array([])
    y_pred = np.array([])
    
    with pytest.raises(ValueError):
        get_regression_metrics(y_pred, y_true)
    
    with pytest.raises(ValueError):
        get_classification_metrics(y_pred, y_true)
    
    with pytest.raises(ValueError):
        bootstrap_metric_ci(y_pred, y_true, lambda y_true, y_pred: 0)

def test_single_value_input():
    """Тест с одним значением"""
    y_true = np.array([1])
    y_pred = np.array([1])
    
    # Для регрессии
    metrics = get_regression_metrics(y_pred, y_true)
    assert np.isclose(metrics['rmse'], 0.0)
    assert np.isclose(metrics['mae'], 0.0)
    assert np.isnan(metrics['r2'])  # R2 не определен для одного значения
    assert np.isclose(metrics['mae_median'], 0.0)
    
    # Для классификации
    metrics = get_classification_metrics(y_pred, y_true)
    assert np.isclose(metrics['accuracy'], 1.0)
    assert np.isclose(metrics['precision'], 1.0)
    assert np.isclose(metrics['recall'], 1.0)
    assert np.isclose(metrics['f1'], 1.0)
    
    # Бутстреп с одним значением
    result = bootstrap_metric_ci(y_pred, y_true, lambda y_true, y_pred: 1, n_bootstrap=10)
    assert result == (1, 1, 1)