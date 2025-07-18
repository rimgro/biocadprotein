# =============================================================================
# prop_prediction/models/embedding_cnn.py
# =============================================================================
# Модуль с архитектурой CNN для работы с эмбеддингами белковых последовательностей
#
# Часть проекта с проектной смены 'Большие Вызовы'
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Основной класс ---

class EmbeddingCNN(nn.Module):
        
    '''
    1-D сверточная сеть для последовательностей, представленных эмбеддингами 
    (в нашем случае 960-мерные вектора из ESM-С).

    Архитектура адаптирована из SequenceCNN (prop_prediction/models/sequence_cnn.py) со следующими изменениями:
        - Удален начальный слой эмбеддингов, так как мы уже получаем эмбеддинги на вход
        - input_channels по умолчанию равен 960 (вместо размера словаря аминокислот)

    Ожидаемая форма входа: [batch_size, seq_len, input_channels]

    Параметры:
        input_channels (int), опц.: размерность входных эмбеддингов (по умолчанию 960)
        num_filters (int), опц.: количество фильтров в сверточных слоях
        dropout_rate (float), опц.: вероятность дропаута

    Примеры:
        >>> from prop_prediction.models.embedding_cnn import EmbeddingCNN
        >>> model = EmbeddingCNN(input_channels=960, num_filters=128)
        >>> output = model(batch_embeddings)  # [batch_size, seq_len, 960]
    '''

    def __init__(
            self,
            input_channels: int = 960,
            num_filters: int = 96,
            dropout_rate: float = 0.4,
        ) -> None:
        super().__init__()

        # Сверточные слои
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_channels, num_filters, kernel_size=5, padding=2)

        # Нормализация
        self.batch_norm1 = nn.BatchNorm1d(num_filters)
        self.batch_norm2 = nn.BatchNorm1d(num_filters)

        # Объединение двух сверточных потоков и уменьшение размерности
        self.conv_merge = nn.Conv1d(num_filters * 2, num_filters, kernel_size=3, padding=1)
        self.batch_norm_merge = nn.BatchNorm1d(num_filters)

        # Глобальный пулинг и полносвязные слои
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters, 256)
        self.fc_out = nn.Linear(256, 1)

        # Регуляризация
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
        '''
        Прямой проход модели.

        Параметры:
            x (torch.Tensor): входной тензор формы [batch_size, seq_len, input_channels]

        Возвращает:
            torch.Tensor: выходной тензор формы [batch_size, 1]
        '''
        
        # Конвертируем в [batch, channels, seq_len] для Conv1d
        x = x.permute(0, 2, 1)

        # Два параллельных сверточных потока
        conv1_out = F.relu(self.batch_norm1(self.conv1(x)))
        conv2_out = F.relu(self.batch_norm2(self.conv2(x)))

        # Объединение и уточнение
        x = torch.cat([conv1_out, conv2_out], dim=1)
        x = F.relu(self.batch_norm_merge(self.conv_merge(x)))
        x = self.dropout(x)

        # Глобальный пулинг и полносвязные слои
        x = self.global_pool(x)
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc_out(x)