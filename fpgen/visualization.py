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
import matplotlib.pyplot as plt

from esm.sdk.api import ESMProtein

def plot_2d_protein(protein: ESMProtein):
    '''
    Строит простую 2D-проекцию белка (XY-плоскость)
    '''
    atoms = protein.to_protein_chain().infer_oxygen().get_atoms()
    
    x = [atom.coord[0] for atom in atoms]
    y = [atom.coord[1] for atom in atoms]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=10, c='green', alpha=0.6)
    plt.title('2D Projection of Protein Structure')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_3d_protein(
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