# =============================================================================
# prop_prediction/models/sequence_cnn.py
# =============================================================================
# Модуль с CNN архитектурой для работы с белковыми последовательностями
#
# Часть проекта с проектной смены 'Большие Вызовы'
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Основной класс ---
    
class SequenceCNN(nn.Module):

    '''
    CNN сеть для анализ белковых последовательностей.
    
    Архитектура:
        - Слой эмбеддингов
        - Два параллельных сверточных слоя с разными размерами ядер (3 и 5)
        - Объединение и дополнительная свертка
        - Глобальный пулинг и полносвязные слои для регрессии
    
    Параметры:
        num_embeddings (int): размер словаря эмбеддингов
        embedding_dim (int), опц.: размерность эмбеддингов (по умолчанию 64)
        num_filters (int), опц.: количество фильтров в сверточных слоях (по умолчанию 96)
    
    Примеры:
        >>> model = SequenceCNN(num_embeddings=100, embedding_dim=64, num_filters=96)
        >>> output = model(batch_indices)  # [batch_size, seq_len]
        >>> print(output.shape)  # [batch_size, 1]
    '''

    def __init__(self, num_embeddings: int, embedding_dim: int = 64, num_filters: int = 96) -> None:
        super(SequenceCNN, self).__init__()

        # Слой эмбеддингов преобразует индексы в плотные векторы
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # Параллельные сверточные слои
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)

        # Нормализация
        self.batch_norm1 = nn.BatchNorm1d(num_filters)
        self.batch_norm2 = nn.BatchNorm1d(num_filters)

        # Объединяющий сверточный слой
        self.conv4 = nn.Conv1d(
            num_filters * 2, num_filters * 1, kernel_size=3, padding=1
        )
        self.batch_norm4 = nn.BatchNorm1d(num_filters * 1)

        # Глобальный пулинг
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Полносвязные слои
        self.fc1 = nn.Linear(num_filters * 1, 256)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''
        Прямой проход модели.
        
        Параметры:
            x (torch.Tensor): входной тензор индексов формы [batch_size, sequence_length]
            
        Возвращает:
            torch.Tensor: выходной тензор формы [batch_size, 1] (регрессионный выход)
        '''

        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]

        conv1_out = F.relu(self.batch_norm1(self.conv1(x)))  # [batch_size, num_filters, seq_len]
        conv2_out = F.relu(self.batch_norm2(self.conv2(x)))  # [batch_size, num_filters, seq_len]

        x = torch.cat([conv1_out, conv2_out], dim=1)  # [batch_size, num_filters*2, seq_len]
        x = F.relu(self.batch_norm4(self.conv4(x)))  # [batch_size, num_filters, seq_len]

        x = self.global_pool(x)  # [batch_size, num_filters, 1]
        x = x.view(x.size(0), -1)  # [batch_size, num_filters]

        x = F.relu(self.fc1(x))  # [batch_size, 256]
        x = self.fc_out(x)  # [batch_size, 1]

        return x