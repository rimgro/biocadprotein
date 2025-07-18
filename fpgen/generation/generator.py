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

from typing import Callable
from typing import Tuple

from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

from fpgen.generation import masking
from fpgen.generation import metrics
from fpgen import utils

# --- Основной класс ---

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
        model: ESM3InferenceClient,
    ) -> None:
        self.__model = model
        self.__protein = protein
        self.__prompt = masking.get_masked_protein(protein, unmasked_indices, self.__model)
    
    def generate(
        self, 
        metric_list: list[str | metrics.Metric | Callable] | None = None,
        temperature: float = 1.0,
        fix_protein: bool = False
    ) -> Tuple[ESMProtein, list[float] | None]:
        
        '''
        Генерирует новый белок на основе базового белка

        Параметры:
            metric_list (list[str | metrics.CustomMetric] | None), опц.:
                Список метрик, которые будут подсчитаны для сгенерированного белка

            temperature (float), опц.:
                Температура генерации новой молекулы

        Возвращемое значение:
            кортеж из двух значений:
                generated_protein (ESMProtein): сгенерированный белок со структурой и последовательностию
                metric_scores (list[float] | None): список с метриками (если переданы) и None в противном случает
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

        # Исправление белка если надо
        proteins = {
            'full_atom': None,
            'backbone': sequence_generation_protein
        }
        
        if fix_protein:
            proteins['full_atom'] = utils.fix_protein(sequence_generation_protein)

        # Если переданы метрики
        if metric_list is not None:
            metric_scores: list[float] = []

            # Проход по каждой метрике
            for metric in metric_list:
                if type(metric) == str:
                    func = metrics.METRIC_NAMES[metric]
                elif callable(metric):
                    func = metric

                # Получение метрики
                key = 'full_atom' if type(metric) == metrics.Metric and metric.calculate_on_full_atom else 'backbone'
                metric_score = func(proteins[key], self.__protein)
                metric_scores.append(metric_score)

            return proteins, metric_scores

        return proteins, None