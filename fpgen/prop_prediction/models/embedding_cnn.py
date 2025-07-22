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
    CNN сеть для анализа предобученных эмбеддингов белковых последовательностей.
    
    Архитектура:
        - Два параллельных сверточных слоя с ядрами 3 и 5
        - Объединение и дополнительная свертка
        - Глобальный пулинг и полносвязные слои
        - Регрессионный выход
    
    Параметры:
        num_filters (int, optional): Количество фильтров в сверточных слоях. По умолчанию 96.
    
    Пример использования:
        >>> model = EmbeddingCNN(num_filters=128)
        >>> embeddings = torch.randn(32, 100, 960)  # [batch, seq_len, embed_dim]
        >>> output = model(embeddings)  # [32, 1]
    '''

    def __init__(self, num_filters: int = 96) -> None:
        super().__init__()

        # Сверточные слои
        self.conv1 = nn.Conv1d(960, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(960, num_filters, kernel_size=5, padding=2)

        # Нормализация
        self.batch_norm1 = nn.BatchNorm1d(num_filters)
        self.batch_norm2 = nn.BatchNorm1d(num_filters)

        # Слой для объединения
        self.conv_merge = nn.Conv1d(
            num_filters * 2, num_filters, kernel_size=3, padding=1
        )
        self.batch_norm_merge = nn.BatchNorm1d(num_filters)

        # Глобальный пулинг
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(num_filters, 256)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''
        Прямой проход модели.
        
        Параметры:
            x (torch.Tensor): входной тензор предобученных эмбеддингов 
                              формы [batch_size, sequence_length, 960]
            
        Возвращает:
            torch.Tensor: выходной тензор формы [batch_size, 1] (регрессионный выход)
        '''

        x = x.permute(0, 2, 1)  # [batch, 960, seq_len]

        conv1_out = F.relu(self.batch_norm1(self.conv1(x)))
        conv2_out = F.relu(self.batch_norm2(self.conv2(x)))

        x = torch.cat([conv1_out, conv2_out], dim=1)
        x = F.relu(self.batch_norm_merge(self.conv_merge(x)))

        x = self.global_pool(x)
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1(x))
        return self.fc_out(x)