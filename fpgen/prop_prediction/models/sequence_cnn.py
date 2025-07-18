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
    1-D сверточная сеть для работы с белковыми последовательностями.
    
    Архитектура:
        - Слой эмбеддингов для аминокислот
        - Два параллельных сверточных потока с разными размерами ядер
        - Объединение признаков и дополнительная свертка
        - Глобальный пулинг и полносвязные слои

    Параметры:
        num_amino_acids (int), опц.: размер словаря аминокислот (по умолчанию 20)
        embedding_dim (int), опц.: размерность эмбеддингов
        num_filters (int), опц.: количество фильтров в сверточных слоях
        dropout_rate (float), опц.: вероятность дропаута

    Примеры:
        >>> from prop_prediction.models.sequence_cnn import SequenceCNN
        >>> model = SequenceCNN(num_amino_acids=20, embedding_dim=64)
        >>> output = model(batch_one_hot)  # [batch_size, seq_len, num_amino_acids]
    '''

    def __init__(
            self,
            num_amino_acids: int = 20,
            embedding_dim: int = 64,
            num_filters: int = 96,
            dropout_rate: float = 0.4,
        ) -> None:
        super().__init__()

        # Слой эмбеддингов для аминокислот
        self.embedding = nn.Embedding(
            num_embeddings=num_amino_acids,
            embedding_dim=embedding_dim
        )

        # Параллельные сверточные потоки
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        
        # Нормализация
        self.batch_norm1 = nn.BatchNorm1d(num_filters)
        self.batch_norm2 = nn.BatchNorm1d(num_filters)

        # Объединение признаков и дополнительная свертка
        self.conv_merge = nn.Conv1d(num_filters * 2, num_filters, kernel_size=3, padding=1)
        self.batch_norm_merge = nn.BatchNorm1d(num_filters)
        
        # Глобальный пулинг и полносвязные слои
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters, 256)
        self.fc_out = nn.Linear(256, 1)
        
        # Регуляризация
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_one_hot: torch.Tensor) -> torch.Tensor:

        '''
        Прямой проход модели.
        
        Параметры:
            x_one_hot (torch.Tensor): входной тензор one-hot векторов 
                                     формы [batch, seq_len, num_amino_acids]
        
        Возвращает:
            torch.Tensor: выходной тензор формы [batch_size, 1]
        '''
        
        # Конвертация one-hot в индексы
        x_indices = torch.argmax(x_one_hot, dim=2)
        
        # Получение эмбеддингов и преобразование формы для Conv1d
        x = self.embedding(x_indices)
        x = x.transpose(1, 2)
        
        # Два параллельных сверточных потока
        conv1_out = F.relu(self.batch_norm1(self.conv1(x)))
        conv2_out = F.relu(self.batch_norm2(self.conv2(x)))
        
        # Объединение и уточнение признаков
        x = torch.cat([conv1_out, conv2_out], dim=1)
        x = F.relu(self.batch_norm_merge(self.conv_merge(x)))
        x = self.dropout(x)
        
        # Глобальный пулинг
        x = self.global_pool(x)
        x = x.flatten(start_dim=1)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        return self.fc_out(x)