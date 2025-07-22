# =============================================================================
# prop_prediction/models/sequence_lstm.py
# =============================================================================
# Модуль с LSTM архитектурой для работы с белковыми последовательностями
#
# Часть проекта с проектной смены 'Большие Вызовы'
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Основной класс ---

class SequenceLSTM(nn.Module):

    '''
    LSTM сеть для анализа белковых последовательностей.
    
    Архитектура:
        - Слой эмбеддингов для преобразования входных индексов в плотные векторы
        - Однослойная LSTM для обработки последовательностей
        - Два полносвязных слоя для регрессии
    
    Параметры:
        num_embeddings (int): размер словаря эмбеддингов
        hidden_dim (int), опц.: размерность скрытого состояния LSTM и эмбеддингов 
                                (по умолчанию 256)
    
    Примеры:
        >>> model = SequenceLSTM(num_embeddings=100, hidden_dim=256)
        >>> output = model(batch_indices)  # [batch_size, seq_len]
    '''
    
    def __init__(self, num_embeddings: int, hidden_dim: int = 256) -> None:
        super(SequenceLSTM, self).__init__()

        self.hidden_dim = hidden_dim

        # Слои эмбеддингов и LSTM
        self.embedding = torch.nn.Embedding(num_embeddings, self.hidden_dim)
        self.lstm = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

        # Полносвязные слои
        self.fc1 = nn.Linear(self.hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        '''
        Прямой проход модели.
        
        Параметры:
            x (torch.Tensor): входной тензор индексов формы [batch_size, sequence_length]
            
        Возвращает:
            torch.Tensor: выходной тензор формы [batch_size, 1] (регрессионный выход)
        '''
    
        x = self.embedding(x)
        x, _ = self.lstm(x)

        x = F.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)

        return x