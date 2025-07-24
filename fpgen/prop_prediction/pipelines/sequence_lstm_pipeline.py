# =============================================================================
# prop_prediction/models/sequence_lstm.py
# =============================================================================
# Модуль с реализацией пайплайна для модели LSTM для работы
# с белковыми последовательностями
#
# Часть проекта с проектной смены 'Большие Вызовы'
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

import torch
import numpy as np
from fpgen.prop_prediction.pipeline import PredictionPipeline
from fpgen.prop_prediction.models.sequence_lstm import SequenceLSTM
from fpgen.prop_prediction.dataset import FPbase
from fpgen.prop_prediction.utils import encode, VOCAB

# --- Основной класс ---

class SequenceLSTMPipeline(PredictionPipeline):

    '''
    Реализация класса PredictionPipeline для использования
    модели LSTM
    '''

    def __init__(self, weight_path: str):
        self.__model = SequenceLSTM(len(VOCAB), hidden_dim=128)
        self.__model.load_state_dict(torch.load(weight_path))
        self.__model.eval()
        self.__fpbase = FPbase()

    def preprocess(self, sequences):
        return torch.tensor([encode(s) for s in sequences], dtype=torch.long)

    def predict(self, inputs):
        with torch.no_grad():
            return self.__model(inputs).squeeze()

    def postprocess(self, raw_preds, target: str):
        preds = raw_preds.numpy()

        if preds.ndim == 0:
            preds = np.array([preds])

        return self.__fpbase.rescale_targets(preds, target).squeeze()
