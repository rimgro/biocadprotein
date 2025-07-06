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
import MDAnalysis as mda

from esm.sdk.api import ESMProtein

def fix_protein(
        protein: ESMProtein | None = None,
        input_filename: str | None = None,
        output_filename: str = 'temp_protein.pdb',
        platform: str = 'CUDA'
    ) -> ESMProtein:

    '''
    Исправляет белок, добавляет полноатомную структуру
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

def get_active_site_residues(
        target_residues: tuple,
        protein: ESMProtein | None = None,
        input_filename: str | None = None,
        output_filename: str = 'temp_protein.pdb',
        radius: float = 5.0
    ) -> list[int]:
    
    '''
    Возвращает индексы остатков в радиусе 'radius' Å от 'target_residues'.
    '''

    if len(target_residues) != 2:
        ValueError('target_residues должен содержать два числа: индекс начала и конца')

    if not ((protein is not None) ^ (input_filename is not None)):
        ValueError('Необходимо передавать либо белок ESMProtein, либо путь к файлу .pdb')

    # Чтение белка
    if protein is not None:
        protein.to_pdb(output_filename)
        input_filename = output_filename

    uni = mda.Universe(input_filename)
    selection = uni.select_atoms(f'around {radius} resid {target_residues[0]}-{target_residues[1]}')
    residues = list(set(selection.residues.resids - 1)) + list(range(target_residues[0] - 1, target_residues[1]))
    
    return list(sorted(residues))