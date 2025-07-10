import numpy as np
import pytest
from fpgen.prop_prediction.metrics import get_regression_metrics, get_classification_metrics

# --- Тесты для регрессии ---

def test_get_regression_metrics_perfect():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = get_regression_metrics(y_pred, y_true)
    
    assert np.isclose(metrics['rmse'], 0.0)
    assert np.isclose(metrics['mae'], 0.0)
    assert np.isclose(metrics['r2'], 1.0)
    assert np.isclose(metrics['mae_median'], 0.0)

def test_get_regression_metrics_nonperfect():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])
    metrics = get_regression_metrics(y_pred, y_true)

    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'mae_median' in metrics
    assert metrics['rmse'] > 0
    assert metrics['mae'] > 0
    assert metrics['r2'] < 1.0

# --- Тесты для классификации ---

def test_get_classification_metrics_perfect():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 2, 1])
    metrics = get_classification_metrics(y_pred, y_true)

    assert np.isclose(metrics['accuracy'], 1.0)
    assert np.isclose(metrics['precision'], 1.0)
    assert np.isclose(metrics['recall'], 1.0)
    assert np.isclose(metrics['f1'], 1.0)

def test_get_classification_metrics_nonperfect():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 2, 2, 0])
    metrics = get_classification_metrics(y_pred, y_true)

    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
