import pytest
import pandas as pd

from fpgen.prop_prediction.dataset import FPbase

@pytest.fixture
def dataset():
    return FPbase()

def test_fpbase_initialization(dataset):
    assert dataset.feature == 'sequence'

    targets = ['brightness', 'em_max', 'ex_max', 'ext_coeff', 'lifetime', 'maturation', 'pka', 'stokes_shift', 'qy', 'agg', 'switch_type']
    assert list(sorted(dataset.targets)) == list(sorted(targets))

    x, y = dataset.get_train('ex_max')
    assert x[0].startswith('MVSKGEELFTGVVPILVEMDGDVNGRKFSVRGVGEGDATHGKLTLKFICTSGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFF')
    
    x, y = dataset.get_test('ex_max')
    assert x[0].startswith('MGSSHHHHHHVSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGD')

def test_preprocess_function():
    proc_dataset = FPbase(preprocess_function=lambda seq: len(seq))
    assert proc_dataset.get_train('ex_max')[0][0] == 239
    assert proc_dataset.get_test('ex_max')[0][0] == 248

def test_scaling_and_rescaling(dataset):
    _, y = dataset.get_train('ex_max', is_scaled=True)
    y_rescaled = dataset.rescale_targets(y, 'ex_max')
    y_original = dataset.get_train('ex_max', is_scaled=False)[1]
    
    # Проверка на то, что масштабирование оратимо
    pd.testing.assert_series_equal(
        pd.Series(y_rescaled.flatten(), name='ex_max'),
        pd.Series(y_original, name='ex_max'),
        check_dtype=False
    )

def test_methods_are_correct(dataset):
    assert len(dataset.to_train_dataframe()) == len(dataset.to_train_dataframe(is_scaled=False))
    assert len(dataset.to_test_dataframe()) == len(dataset.to_test_dataframe(is_scaled=False))
    assert len(dataset.to_train_dataframe()) != len(dataset.to_test_dataframe())