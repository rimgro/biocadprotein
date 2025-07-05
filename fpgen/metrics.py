# =============================================================================
# metrics.py
# =============================================================================
# Метрики качества сгенерированных белков.
# Включает RMSD (сравнение со структурным шаблоном), pLDDT и pTM score.
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

from typing import Callable, Union

import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics

from esm.sdk import client
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.models.esm3 import ESM3
from esm.utils.structure.protein_chain import ProteinChain

def rmsd(generation_protein: ESMProtein, template_protein: ESMProtein) -> float:

    '''
    Вычисляет метрику RMSD — это мера структурного расстояния между координатами
    '''

    template_chain: ProteinChain = template_protein.to_protein_chain()
    generation_chain: ProteinChain = generation_protein.to_protein_chain()

    rmsd = template_chain.rmsd(generation_chain)
    return rmsd

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

def identity(generation_protein: ESMProtein, template_protein: ESMProtein) -> float:

    '''
    Вычисляет метрику похожести молекул
    '''

    seq1 = seq.ProteinSequence(template_protein.sequence)
    seq2 = seq.ProteinSequence(generation_protein.sequence)

    alignments = align.align_optimal(
        seq1, seq2, align.SubstitutionMatrix.std_protein_matrix(), gap_penalty=(-10, -1)
    )

    alignment = alignments[0]
    identity = align.get_sequence_identity(alignment)
    return identity

METRIC_NAMES = {
    'rmsd': rmsd,
    'ptm': ptm,
    'plddt': plddt,
    'identity': identity
}

class Metric:
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