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

from esm.sdk import client
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.models.esm3 import ESM3
from esm.utils.structure.protein_chain import ProteinChain

def rmsd(generation_protein: ESMProtein, template_protein: ESMProtein) -> float:
    template_chain: ProteinChain = template_protein.to_protein_chain()
    generation_chain: ProteinChain = generation_protein.to_protein_chain()

    rmsd = template_chain.rmsd(generation_chain)
    return rmsd

def ptm(generation_protein: ESMProtein, *args, **kwargs) -> float:
    return generation_protein.ptm.item()

def plddt(generation_protein: ESMProtein, *args, **kwargs) -> float:
    return generation_protein.plddt.median().item()

METRIC_NAMES = {
    'rmsd': rmsd,
    'ptm': ptm,
    'plddt': ptm
}

class CustomMetric:
    def __init__(self, metric_func: callable, *args, **kwargs):
        self.__func = metric_func
        self.__args = args
        self.__kwargs = kwargs

    def __call__(self, generation_protein: ESMProtein, template_protein: ESMProtein):
        # При вызове объединяем init и call параметры
        return self.__func(
            generation_protein,
            template_protein,
            *self.__args,
            **self.__kwargs
        )