# =============================================================================
# prop_prediction/models/embedding_lstm.py
# =============================================================================
# Модуль с архитектурой LSTM для работы с эмбеддингами белковых последовательностей
#
# Часть проекта с проектной смены 'Большие Вызовы'
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Основной класс ---

class EmbeddingLSTM(nn.Module):

    '''
    LSTM сеть для анализа предобученных эмбеддингов белковых последовательностей.
    
    Архитектура:
        - Линейная проекция входных эмбеддингов в пространство меньшей размерности
        - Однослойная LSTM для обработки последовательностей
        - Два полносвязных слоя с усреднением по последовательности
        - Выходной слой для регрессии
    
    Параметры:
        hidden_size (int): размерность скрытого состояния LSTM и проекционного слоя
    
    Примеры:
        >>> model = EmbeddingLSTM(hidden_size=256)
        >>> output = model(precomputed_embeddings)  # [batch_size, seq_len, 960]
        >>> print(output.shape)  # [batch_size, 1]
    '''

    def __init__(self, hidden_size) -> None:
        super(EmbeddingLSTM, self).__init__()

        # Фиксированная размерность входных эмбеддингов
        self.embed_dim = 960
        self.hidden_size = hidden_size

        # Проекционный слой для уменьшения размерности эмбеддингов
        self.input_projection = nn.Linear(self.embed_dim, self.hidden_size)
        
        # Рекуррентный слой
        self.lstm = torch.nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True
        )
        
        # Полносвязные слои
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''
        Прямой проход модели.
        
        Параметры:
            x (torch.Tensor): входной тензор предобученных эмбеддингов 
                              формы [batch_size, sequence_length, 960]
            
        Возвращает:
            torch.Tensor: выходной тензор формы [batch_size, 1] (регрессионный выход)
        '''

        x = F.relu(self.input_projection(x))  # [batch_size, seq_len, hidden_size]
        x, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size]

        x = F.relu(self.fc1(x))  # [batch_size, seq_len, 64]
        x = self.fc2(x)  # [batch_size, seq_len, 1]
        
        x = torch.mean(x, dim=1)  # [batch_size, 1]

        return x