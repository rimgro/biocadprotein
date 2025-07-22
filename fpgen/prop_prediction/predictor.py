# =============================================================================
# prop_prediction/predictor.py
# =============================================================================
# Главный интерфейс предсказания свойств белков.
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

import numpy as np
import torch

from fpgen.prop_prediction.models import (
    EmbeddingCNN,
    EmbeddingLSTM,
    SequenceLSTM,
    SequenceCNN
)
from fpgen.prop_prediction.dataset import FPbase

VOCAB = {char: (i + 1) for i, char in enumerate('ACDEFGHIKLMNPRQSTVWYX')}
PAD_TOKEN = '?'
VOCAB[PAD_TOKEN] = 0
VOCAB_REV = {v: k for k, v in VOCAB.items()}

def encode(sequence: str) -> list[int]:
    if sequence is None:
        return sequence
    
    encode_list = []
    for s in sequence:
        if s in VOCAB:
            encode_list.append(VOCAB[s])
        else:
            encode_list.append(VOCAB['X'])
    return encode_list

class PropertiesPredictor:
    def __init__(self, model_name: str = 'sequence-cnn-full'):
        self.model_name = model_name
        self.fpbase = FPbase()

    def __call__(self, x: list[str] | str, target: str):
        if isinstance(x, str):
            x = [x]

        x_lst = [encode(i) for i in x]

        model = SequenceCNN(len(VOCAB), embedding_dim=128, num_filters=64)
        model.load_state_dict(torch.load(f'weights/{self.model_name}.pth'))

        y = model(torch.tensor(x_lst, dtype=torch.long)).squeeze()
        
        if y.numel() == 1:
            y = torch.tensor([y])

        y_rescaled = self.fpbase.rescale_targets(y.detach().numpy(), target)

        return y_rescaled.squeeze()