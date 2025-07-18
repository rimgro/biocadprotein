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
    LSTM-сеть для анализа белковых последовательностей, представленных эмбеддингами.
    Поддерживает двунаправленную обработку последовательностей и несколько слоев LSTM.
    
    Ожидаемая форма входа: [batch_size, seq_len, embed_dim]
    
    Параметры:
        vocab_size (int), опц.: размер словаря (21 для 20 аминокислот + padding)
        embed_dim (int), опц.: размерность входных эмбеддингов (по умолчанию 960)
        hidden_size (int), опц.: количество нейронов в LSTM слоях
        num_layers (int), опц.: количество LSTM слоев
        dropout_rate (float), опц.: вероятность дропаута
        
    Примеры:
        >>> from prop_prediction.models.embedding_lstm import EmbeddingLSTM
        >>> model = EmbeddingLSTM(embed_dim=960, hidden_size=256)
        >>> output = model(batch_embeddings)  # [batch_size, seq_len, 960]
    '''

    def __init__(
            self,
            vocab_size: int = 21,
            embed_dim: int = 960,
            hidden_size: int = 256,
            num_layers: int = 3,
            dropout_rate: float = 0.5
        ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        # Проекция входных эмбеддингов
        self.input_projection = nn.Linear(embed_dim, hidden_size // 2)
        
        # Двунаправленные LSTM слои
        self.lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Выходные слои
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # Регуляризация
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
        '''
        Прямой проход модели.
        
        Параметры:
            x (torch.Tensor): входной тензор формы [batch_size, seq_len, embed_dim]
            
        Возвращает:
            torch.Tensor: выходной тензор формы [batch_size]
        '''
        
        # Создаем маску для padding токенов
        mask = (x != 0).float()  # 1 для реальных токенов, 0 для padding
        mask = mask[:, :, 0].unsqueeze(-1)
        
        # Применяем маску к эмбеддингам
        x = x * mask
        
        # Проекция входа в скрытое пространство
        x = F.relu(self.input_projection(x))
        
        # Обработка LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Применяем маску к выходу LSTM
        lstm_out = lstm_out * mask
        
        # Нормализация
        lstm_out = self.layer_norm(lstm_out)
        
        # Выходные слои
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        # Усреднение по последовательности
        x = torch.mean(x, dim=1)
        
        # Удаляем последнюю размерность
        return x.squeeze(-1)