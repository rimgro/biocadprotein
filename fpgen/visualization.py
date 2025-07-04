# =============================================================================
# visualization.py
# =============================================================================
# Простые функции для визуализации масок и последовательностей.
# Используется для отладки и анализа результата.
#
# Часть проекта с проектной смены "Большие Вызовы"
# 
# Лицензия: MIT (см. LICENSE)
# =============================================================================

import py3Dmol
from esm.sdk.api import ESMProtein

def plot_protein(
        protein: ESMProtein,
        width: int = 1000,
        height: int = 500,
        color: str = 'lightgreen'
    ) -> None:

    '''
    Строит 3D модель белка
    '''

    view = py3Dmol.view(width=width, height=height)
    view.addModel(
        protein.to_protein_chain().infer_oxygen().to_pdb_string(),
        'pdb',
    )
    view.setStyle({'cartoon': {'color': color}})
    view.zoomTo()
    view.show()