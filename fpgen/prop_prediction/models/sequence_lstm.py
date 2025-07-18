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
    LSTM сеть для анализа белковых последовательностей, представленных one-hot векторами.
    
    Архитектура:
        - Проекция входных one-hot векторов в пространство меньшей размерности
        - Стек двунаправленных LSTM слоев
        - Слой нормализации
        - Полносвязные слои для регрессии

    Параметры:
        input_dim (int), опц.: размерность входных one-hot векторов (по умолчанию 20)
        hidden_dim (int), опц.: размерность скрытого состояния LSTM
        num_layers (int), опц.: количество LSTM слоев
        output_dim (int), опц.: размерность выхода (по умолчанию 1 для регрессии)
        dropout (float), опц.: вероятность дропаута
        bidirectional (bool), опц.: использовать двунаправленную LSTM

    Примеры:
        >>> from prop_prediction.models.sequence_lstm import SequenceLSTM
        >>> model = SequenceLSTM(input_dim=20, hidden_dim=256)
        >>> output = model(batch_one_hot)  # [batch_size, seq_len, input_dim]
    '''

    def __init__(
            self,
            input_dim: int = 20,
            hidden_dim: int = 256,
            num_layers: int = 2,
            output_dim: int = 1,
            dropout: float = 0.3,
            bidirectional: bool = True,
        ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Проекция one-hot векторов в пространство меньшей размерности
        self.input_projection = nn.Linear(input_dim, hidden_dim // 2)
        
        # Стек LSTM слоев
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Выходные слои
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc_out = nn.Linear(128, output_dim)
        
        # Регуляризация
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Прямой проход модели.
        
        Параметры:
            x (torch.Tensor): входной тензор one-hot векторов 
                              формы [batch_size, sequence_length, input_dim]
            
        Возвращает:
            torch.Tensor: выходной тензор формы [batch_size, output_dim]
        '''
        # Проекция входных векторов
        x = F.relu(self.input_projection(x))
        
        # Обработка LSTM
        lstm_out, _ = self.lstm(x)
        
        # Нормализация и взятие последнего скрытого состояния
        lstm_out = self.layer_norm(lstm_out[:, -1, :])
        
        # Полносвязные слои
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.fc_out(x)
        
        return x