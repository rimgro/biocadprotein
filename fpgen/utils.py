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
from openmm import Platform

from esm.sdk.api import ESMProtein

def fix_protein(protein: ESMProtein, filename: str = 'temp_protein.pdb') -> ESMProtein:

    '''
    Исправляет белок
    '''

    root_dir = os.getcwd()
    filename = os.path.join(root_dir, filename)
    protein.to_pdb(filename)

    platform = Platform.getPlatformByName('CUDA')
    fixer = PDBFixer(filename=filename, platform=platform)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    PDBFile.writeFile(fixer.topology, fixer.positions, open(filename, 'w'))
    fixed_protein = ESMProtein.from_pdb(filename)

    return fixed_protein
