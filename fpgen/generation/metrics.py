# =============================================================================
# generation/metrics.py
# =============================================================================
# Метрики качества сгенерированных белков.
# Включает RMSD (сравнение со структурным шаблоном), pLDDT и pTM score.
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

from typing import Callable

import biotite.sequence as seq
import biotite.sequence.align as align

from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain

# --- Метрики для сравнения двух белковых молекуд ---

def rmsd(generation_protein: ESMProtein, template_protein: ESMProtein) -> float:
    '''
    Вычисляет метрику RMSD — это мера структурного расстояния между координатами
    '''
    template_chain: ProteinChain = template_protein.to_protein_chain()
    generation_chain: ProteinChain = generation_protein.to_protein_chain()

    rmsd = template_chain.rmsd(generation_chain)
    return rmsd

def identity(generation_protein_seq: ESMProtein | str, template_protein_seq: ESMProtein | str) -> float:

    '''
    Вычисляет метрику похожести молекул
    '''

    # Если generation_protein это белок ESM3, то извлекаем последовательность аминокислот
    if hasattr(generation_protein_seq, 'sequence'):
        generation_protein_seq = generation_protein_seq.sequence

    if hasattr(template_protein_seq, 'sequence'):
        template_protein_seq = template_protein_seq.sequence

    # Обертка в класс последовательности
    seq1 = seq.ProteinSequence(template_protein_seq)
    seq2 = seq.ProteinSequence(generation_protein_seq)
    
    # Выравнивание
    alignments = align.align_optimal(
        seq1,
        seq2,
        align.SubstitutionMatrix.std_protein_matrix(),
        gap_penalty=(-10, -1),
        terminal_penalty=False
    )

    alignment = alignments[0]

    # Получение метрики
    identity = align.get_sequence_identity(alignment)
    return identity

# --- Метрики для метрик существования белков ---

def ptm(generation_protein: ESMProtein, *args, **kwargs) -> float:
    '''
    Вычисляет метрику PTM
    '''
    return generation_protein.ptm.item()

def plddt(generation_protein: ESMProtein, *args, **kwargs) -> float:
    '''
    Вычисляет метрику pLDDT (predicted Local Distance Difference Test) — метрика уверенности модели AlphaFold в точности предсказания локальной структуры (чем выше, тем точнее, диапазон 0–100).
    '''
    return generation_protein.plddt.mean().item()


# --- Словарь со всеми метриками ---

METRIC_NAMES = {
    'rmsd': rmsd,
    'ptm': ptm,
    'plddt': plddt,
    'identity': identity
}

# --- Абстрактный класс для метрик ---

class Metric:

    '''
    Абстрактный класс для представления метрики

    Параметры:
        metric_func (Callable | str):
            Функция (либо ее название, если используете стандартные метрики)

        Может принимать дополнительные параметры, которые будут переданы в функцию метрики

    Примеры:
        >>> from fpgen.generation.metrics import Metric

        >>> protein = ...
        >>> ptm_metric = Metric('ptm')
        >>> ptm_metric(protein)

        >>> len_metric = Metric(lambda x: len(x.sequence))
        >>> len_metric(protein)
    '''

    def __init__(
        self,
        metric_func: Callable | str,
        calculate_on_full_atom: bool = False,
        *args,
        **kwargs
    ):
        if type(metric_func) == str:
            self.__func = METRIC_NAMES[metric_func]
        elif callable(metric_func):
            self.__func = metric_func

        self.calculate_on_full_atom = calculate_on_full_atom
        self.__args = args
        self.__kwargs = kwargs
    
    def __call__(self, generation_protein: ESMProtein, template_protein: ESMProtein):
        return self.__func(
            generation_protein,
            template_protein,
            *self.__args,
            **self.__kwargs
        )