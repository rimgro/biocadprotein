import pytest
import pandas as pd

from fpgen.prop_prediction.dataset import FPbase

def test_fpbase_initialization():
    dataset = FPbase()
    assert dataset.feature == 'sequence'

    targets = ['brightness', 'em_max', 'ex_max', 'ext_coeff', 'lifetime', 'maturation', 'pka', 'stokes_shift', 'qy', 'agg']
    assert list(sorted(dataset.targets)) == list(sorted(targets))

    # Проверка на паттерн синглтон
    dataset2 = FPbase()
    assert dataset is dataset2

def test_preprocess_function():
    dataset = FPbase(preprocess_function=lambda seq: len(seq))

def test_scaling_and_rescaling():
    dataset = FPbase()

    x, y = dataset.get_train('ex_max', is_scaled=True)
    y_rescaled = dataset.rescale_targets(y, 'ex_max')
    y_original = dataset.get_train('ex_max', is_scaled=False)[1]
    
    # Проверка на то, что масштабирование оратимо
    pd.testing.assert_series_equal(
        pd.Series(y_rescaled.flatten(), index=y_original.index, name='ex_max'),
        y_original,
        check_dtype=False
    )

def test_methods_are_correct():
    dataset = FPbase()

    assert len(dataset.to_train_dataframe()) == len(dataset.to_train_dataframe(is_scaled=False))
    assert len(dataset.to_test_dataframe()) == len(dataset.to_test_dataframe(is_scaled=False))
    assert len(dataset.to_train_dataframe()) != len(dataset.to_test_dataframe())