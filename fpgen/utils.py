# =============================================================================
# utils.py
# =============================================================================
# Дополнительные утилиты
#
# Часть проекта с проектной смены "Большие Вызовы"
# Лицензия: MIT (см. LICENSE)
# =============================================================================

import os

from pdbfixer import PDBFixer
from openmm.app import PDBFile

from esm.sdk.api import ESMProtein

def fix_protein(protein: ESMProtein, filename: str = 'temp_protein.pdb') -> ESMProtein:

    '''
    Исправляет белок
    '''

    print(1)
    root_dir = os.getcwd()
    filename = os.path.join(root_dir, filename)
    protein.to_pdb(filename)

    print(2)
    fixer = PDBFixer(filename=filename)
    print(3)
    fixer.findMissingResidues()
    print(4)
    fixer.findMissingAtoms()
    print(5)
    fixer.addMissingAtoms()
    print(6)
    fixer.addMissingHydrogens(pH=7.0)

    PDBFile.writeFile(fixer.topology, fixer.positions, open(filename, 'w'))
    fixed_protein = ESMProtein.from_pdb(filename)

    return fixed_protein
