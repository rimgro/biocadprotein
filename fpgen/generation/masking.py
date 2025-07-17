# =============================================================================
# generation/masking.py
# =============================================================================
# Маскирование белка
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

import torch
from esm.sdk import client
from esm.sdk.api import ESM3InferenceClient, ESMProtein

def get_masked_protein(
        protein: ESMProtein,
        unmasked_indices: list[int],
        model: ESM3InferenceClient,
    ) -> ESMProtein:


    '''
    Маскирует белок

    Параметры:
        protein (ESMProtein): белок
        unmasked_indices (list[int]): индексы аминокислот, которые не будут маскироваться
        model (ESM3InferenceClient): модель ESM3, для которой нужно произвести подготовку данных
    '''
    
    with torch.no_grad():
        # Токенизация
        protein_tokens = model.encode(protein)

        # Замаскированная последовательность
        prompt_seq = ['_'] * len(protein.sequence)
        
        # Убираем маску там, где находится альфа спираль
        for i in unmasked_indices:
            prompt_seq[i] = protein.sequence[i]

        prompt_seq = ''.join(prompt_seq)
        prompt = model.encode(ESMProtein(sequence=prompt_seq))
        prompt.structure = torch.full_like(prompt.sequence, 4096)

        # Добавляем начальный и конечный токены
        prompt.structure[0] = 4098
        prompt.structure[-1] = 4097

        # Заполняем токены для промпта
        for i in unmasked_indices:
            prompt.structure[i + 1] = protein_tokens.structure[i + 1]

        return prompt