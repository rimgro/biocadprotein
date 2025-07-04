# =============================================================================
# generator.py
# =============================================================================
# Главный интерфейс генерации белков с помощью ESM3.
# Обрабатывает входные белки и маски, вызывает модель,
# возвращает результат в удобной форме.
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

from esm.sdk import client
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.models.esm3 import ESM3
from esm.utils.structure.protein_chain import ProteinChain

from fpgen import masking
from fpgen import metrics

class ProteinGenerator:

    '''
    Класс для представления генератора новых вариантов белков на основе
    заданного белка с использованием модели ESM3.
    
    Параметры:
        protein (ESMProtein): Исходный белок, используемый для генерации
        unmasked_indices (list[int]):
            Индексы токенов в последовательности и структуре, которые не будут
            маскироваться
        model (ESM3InferenceClient): Клиент для работы с моделью ESM3
    
    Методы:
        generate(...): Генерирует новый вариант белка и при необходимости вычисляет метрики
    '''

    def __init__(
        self,
        protein: ESMProtein, 
        unmasked_indices: list[int],
        model: ESM3InferenceClient
    ) -> None:
        self.__model = model
        self.__protein = protein
        self.__prompt = masking.get_masked_protein(protein, unmasked_indices, self.__model)
    
    def generate(
        self, 
        metric_list: list[str | metrics.CustomMetric] | None = None,
        temperature: float = 1.0
    ) -> ESMProtein:
        
        '''
        Генерирует новый белок на основе базового белка

        Параметры:
            metric_list (list[str | metrics.CustomMetric] | None), опц.:
                Список метрик, которые будут подсчитаны для сгенерированного белка

            temperature (float), опц.:
                Температура генерации новой молекулы
        '''

        # Количество токенов, которые нужно предсказать
        num_tokens_to_decode = 20

        # Генерация новой структуры на основе промпта
        structure_generation = self.__model.generate(
            self.__prompt,
            GenerationConfig(
                track='structure',
                num_steps=num_tokens_to_decode,
                temperature=temperature,
            ),
        )

        # Генерация последовательности на основе структуры
        sequence_generation = self.__model.generate(
            structure_generation,
            GenerationConfig(track='sequence', num_steps=num_tokens_to_decode),
        )

        # Декодирование последовательности в строку и структуры в координаты
        sequence_generation_protein = self.__model.decode(sequence_generation)

        # Если переданы метрики
        if metric_list is not None:
            metric_scores: list[float] = []

            # Проход по каждой метрике
            for metric in metric_list:
                if type(metric) == str:
                    func = metrics.METRIC_NAMES[metric]
                else:
                    func = metric

                # Получение метрики
                metric_score = func(sequence_generation_protein, self.__protein)
                metric_scores.append(metric_score)

            return sequence_generation_protein, metric_scores

        return sequence_generation_protein