# =============================================================================
# prop_prediction/registry.py
# =============================================================================
# Модуль с реестром моделей
#
# Часть проекта с проектной смены 'Большие Вызовы'
# Лицензия: MIT (см. LICENSE)
# =============================================================================

from fpgen.prop_prediction.pipeline import PredictionPipeline
from fpgen.prop_prediction.pipelines.sequence_lstm_pipeline import SequenceLSTMPipeline
from fpgen.prop_prediction.pipelines.sequence_cnn_pipeline import SequenceCNNPipeline

_PIPELINES = {
    'sequence-cnn-full': lambda: SequenceCNNPipeline('weights/sequence-cnn-full.pth'),
    'sequence-lstm-full': lambda: SequenceLSTMPipeline('weights/sequence-lstm-full.pth')
}

def get_pipeline(name: str) -> PredictionPipeline:

    '''
    Возвращает объект пайплайна для предсказания спектральных свойств

    Параметры:
        name (str): Название модели
    
    Возвращает:
        PredictionPipeline: Реализация пайплайна
    '''

    if name not in _PIPELINES:
        raise ValueError(f'Неизвестное имя модели: {name}. Доступны только: {", ".join(_PIPELINES)}')
    
    return _PIPELINES[name]()
