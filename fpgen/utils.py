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

def fix_protein(
        protein: ESMProtein | None = None,
        input_filename: str | None = None,
        output_filename: str = 'temp_protein.pdb',
        platform: str = 'CUDA'
    ) -> ESMProtein:

    '''
    Исправляет белок
    '''

    root_dir = os.getcwd()
    output_path = os.path.join(root_dir, output_filename)

    if protein is not None:
        protein.to_pdb(output_path)
        input_filename = output_path

    platform = Platform.getPlatformByName(platform)
    fixer = PDBFixer(filename=input_filename, platform=platform)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_path, 'w'))
    fixed_protein = ESMProtein.from_pdb(output_path)

    return fixed_protein
