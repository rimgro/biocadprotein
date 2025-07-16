# =============================================================================
# utils.py
# =============================================================================
# Дополнительные утилиты (в основном для работы с PDB файлами и белками)
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

# --- Вспомогательные функции ---

def fix_protein(
        protein: ESMProtein | None = None,
        input_filename: str | None = None,
        output_filename: str = 'temp_protein.pdb',
        platform: str = 'CUDA'
    ) -> ESMProtein:

    '''
    Исправляет белок, добавляет полноатомную структуру

    Параметры:
        protein (ESMProtein) или input_filename (str): белок или путь к PDB файлу белка
        output_filename (str), опц.: имя выходного временного PDB файла
        platform (str), опц.: CPU или CUDA
    '''

    # Путь для сохранения временного PDB файла
    root_dir = os.getcwd()
    output_path = os.path.join(root_dir, output_filename)

    # Если передан белок, а не input_filename, то сохранение в формате PDB
    if protein is not None:
        protein.to_pdb(output_path)
        input_filename = output_path

    # Исправление PDB
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
    Возвращает индексы аминокислот в радиусе 'radius' Å от 'target_residues'.

    Параметры:
        target_residues: от какого до какого индекса считать активным центром (нумерация от 1)
        protein (ESMProtein) или input_filename (str): белок или путь к PDB файлу белка
        output_filename (str), опц.: имя выходного временного PDB файла
        radius (float), опц.: расстояние от активного центра, до которого считается поддерживающей оболочкой (в ангстремах)

    Возвращает:
        Список из индексов (нумерация с 0) аминокислот, которые считаются поддерживающим центром
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

    start, end = target_residues
    # Выбор поддерживающего центра на расстоянии radius
    selection = uni.select_atoms(f'around {radius} resid {start}-{end}')
    # Объединение с активным центром, перевод в 0-base
    residues = list(set(selection.residues.resids)) + list(range(start, end + 1))

    # В PDB файле некоторые аминокислоты могут отсутствовать
    # Например: 74, 75, _, _, 78. Тогда индексация может нарушиться
    valid_residue_indices = []
    sequence_index = 0
    for residue in uni.select_atoms('protein').residues:
        # Если есть CA атом — считаем, что аминокислота есть
        if any(atom.name == 'CA' for atom in residue.atoms):
            valid_residue_indices.append((residue.resid, sequence_index))
            sequence_index += 1
        else:
            continue

    # Создаём соответствие: resid -> индекс в protein.sequence
    residue_map = dict(valid_residue_indices)
    residues = [residue_map[i] for i in residues if i in residue_map]
    
    return list(sorted(residues))

def pdb_to_uni_mol(input_filename: str) -> dict:
    '''
    Преобразовывает PDB в формат для модели Uni-Mol
    
    Параметры:
        input_filename (str): путь к PDB файлу

    Пример:
        >>> from fpgen.utils import pdb_to_uni_mol
        >>> data = pdb_to_uni_mol('structure.pdb')

    Возвращает:
        Словарь для подачи в модель Uni-Mol

    Пример возвращаемого словаря:
        {
            'atoms': ['C', 'H', 'O'],
            'coordinates': [
                (20.0, 20.0, 20.0), (21.0, 20.0, 20.0), (20.0, 21.0, 20.0)
            ]
        }
    '''
    with open(input_filename) as f:
        pdb_lines = f.readlines()

    result = {'atoms': [], 'coordinates': []}

    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_type = line[76:78].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            result['atoms'].append(atom_type)
            result['coordinates'].append((x, y, z))

    return result