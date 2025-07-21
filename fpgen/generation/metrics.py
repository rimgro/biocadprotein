# =============================================================================
# generation/metrics.py
# =============================================================================
# Метрики качества сгенерированных белков.
# Включает RMSD (сравнение со структурным шаблоном), pLDDT и pTM score.
#
# Часть проекта с проектной смены 'Большие Вызовы'
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

from typing import Callable

import numpy as np

import biotite.sequence as seq
import biotite.sequence.align as align

from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain

from fpgen import utils

# --- Приватные функции (используются только в метриках)

def _get_sequence_weighted_identity(
        alignment,
        weighted_idx: list[int],
        weight: float = 2.0,
        mode: str = 'not_terminal'
    ) -> float:

    '''
    Вычисляет взвешенную идентичность последовательностей для выравнивания.

    Идентичность рассчитывается как количество совпадений, деленное на меру
    длины выравнивания, которая зависит от параметра `mode`. Для позиций,
    указанных в `weighted_idx`, совпадения учитываются с дополнительным весом.

    Параметры:
        alignment (Alignment): выравнивание для расчета идентичности
        weighted_idx (list[int]): список индексов позиций с дополнительным весом
        weight (float), опц.: вес для указанных позиций (по умолчанию 2.0)
        mode (str),: режим расчета длины выравнивания:
            - 'all' - количество совпадений делится на общее число колонок выравнивания
            - 'not_terminal' - исключаются терминальные гэпы (по умолчанию)
            - 'shortest' - делится на длину самой короткой последовательности

    Возвращает:
        float: значение идентичности последовательностей в диапазоне от 0 до x

    Исключения:
        ValueError: если выбран недопустимый режим или нет перекрытия последовательностей

    Примеры:
        >>> identity = get_sequence_weighted_identity(alignment, [10, 20, 30], 2.0)
    '''

    codes = align.get_codes(alignment)

    # Подсчет совпадений с учетом весов
    matches = 0
    for i in range(codes.shape[1]):
        column = codes[:, i]
        unique_symbols = np.unique(column)
        
        # Все символы в колонке совпадают и не являются гэпами
        if len(unique_symbols) == 1 and unique_symbols[0] != -1:
            matches += (weight if i in weighted_idx else 1.0)

    # Расчет длины выравнивания в зависимости от режима
    if mode == 'all':
        length = len(alignment)
    elif mode == 'not_terminal':
        start, stop = align.find_terminal_gaps(alignment)
        if stop <= start:
            raise ValueError(
                'Невозможно рассчитать идентичность - '
                'последовательности не перекрываются'
            )
        length = stop - start
    elif mode == 'shortest':
        length = min(len(seq) for seq in alignment.sequences)
    else:
        raise ValueError(f"Недопустимый режим расчета: '{mode}'")

    return matches / length

# --- Метрики для сравнения двух белковых молекуд ---

def rmsd(generation_protein: ESMProtein, template_protein: ESMProtein) -> float:
    '''
    Вычисляет метрику RMSD — это мера структурного расстояния между координатами
    '''
    template_chain: ProteinChain = template_protein.to_protein_chain()
    generation_chain: ProteinChain = generation_protein.to_protein_chain()

    rmsd = template_chain.rmsd(generation_chain)
    return rmsd

def seq_identity(generation_protein_seq: ESMProtein | str, template_protein_seq: ESMProtein | str) -> float:

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
    )

    alignment = alignments[0]

    # Получение метрики
    identity = align.get_sequence_identity(alignment)
    return identity

def weighted_seq_idenity(
        seq1: ESMProtein | str,
        seq2: ESMProtein | str,
        weighted_idx: list[int],
        weight: float = 2.0
    ) -> float:

    # Если generation_protein это белок ESM3, то извлекаем последовательность аминокислот
    if hasattr(seq1, 'sequence'):
        seq1 = seq1.sequence

    if hasattr(seq2, 'sequence'):
        seq2 = seq2.sequence

    # Обертка в класс последовательности
    seq1 = seq.ProteinSequence(seq1)
    seq2 = seq.ProteinSequence(seq2)
    
    # Выравнивание
    alignments = align.align_optimal(
        seq1,
        seq2,
        align.SubstitutionMatrix.std_protein_matrix(),
    )

    alignment = alignments[0]

    # array([[ 0, 20,  1, -1, -1, -1],
    #    [ 0, 20,  1,  1,  1,  1]], dtype=int64)
    codes = align.get_codes(alignment)

    # [ 0, 20,  1, -1, -1, -1]
    seq1_code = codes[0]

    # Новые индексы активного центра с учетом выравнивания
    aligned_weighted_idx = utils.get_align_alpha_phelix_idx(seq1_code, weighted_idx)

    # Получение метрики
    identity = _get_sequence_weighted_identity(alignment, aligned_weighted_idx, weight)
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
    'seq_identity': seq_identity,
    'weighted_seq_identity': weighted_seq_idenity
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