# =============================================================================
# utils.py
# =============================================================================
# Дополнительные утилиты (в основном для работы с PDB файлами и белками)
#
# Часть проекта с проектной смены 'Большие Вызовы'
# Лицензия: MIT (см. LICENSE)
# =============================================================================

import os

from pdbfixer import PDBFixer
from openmm.app import PDBFile
from openmm import Platform
import MDAnalysis as mda

import biotite.sequence as seq
import biotite.sequence.align as align

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

# --- Функции для работы с активным центром ---

import MDAnalysis as mda

def get_active_site_residues(
        target_residues: tuple[int, int],
        protein: ESMProtein | None = None,
        input_filename: str | None = None,
        output_filename: str = 'temp_protein.pdb',
        radius: float = 5.0
    ) -> list[int]:
    
    '''
    Возвращает индексы (0-based) аминокислот в радиусе 'radius' Å от остатков
    с номерами от target_residues[0] до target_residues[1] (включительно).

    Параметры:
        target_residues: кортеж (start, end) номеров остатков (1-based)
        protein: ESMProtein или None
        input_filename: путь к PDB файлу или None
        output_filename: временный PDB, если передан protein
        radius: радиус поиска (Å)

    Возвращает:
        Отсортированный список 0-based индексов остатков, входящих в активную зону.
    '''

    # Проверка входных аргументов
    if len(target_residues) != 2:
        raise ValueError('target_residues должен содержать два числа: начало и конец (1-based)')
    if not ((protein is None) ^ (input_filename is None)):
        raise ValueError('Нужно передать либо protein, либо input_filename, но не оба одновременно')

    if protein is not None:
        protein.to_pdb(output_filename)
        input_filename = output_filename

    uni = mda.Universe(input_filename)
    
    # 1) Собираем список всех остатков и строим маппинг:
    all_res = uni.select_atoms('protein').residues
    seq_to_pdb: dict[int, tuple[int, str]] = {}
    for seq_idx, res in enumerate(all_res):
        # вместо res.segments.segids[0] используем res.segid
        seq_to_pdb[seq_idx] = (res.resid, res.segid)

    # 2) Переводим 1-based → 0-based и корректируем порядок
    start0 = target_residues[0] - 1
    end0   = target_residues[1] - 1
    if start0 > end0:
        start0, end0 = end0, start0

    # 3) Собираем множество (resid, chain) целевой области
    target_pdb = {
        seq_to_pdb[i]
        for i in range(start0, end0 + 1)
        if i in seq_to_pdb
    }
    if not target_pdb:
        return []

    # 4) Жёстко фиксируем поиск по первой из цепей
    _, chain_id = next(iter(target_pdb))
    sel_targets = uni.select_atoms(
        "protein and segid {} and resid {}".format(
            chain_id,
            " ".join(str(r) for r, _ in target_pdb)
        )
    )

    # 5) Находим «around radius» от целевой группы
    sel_neighbors = uni.select_atoms(
        f"around {radius} group targ", targ=sel_targets
    )

    # 6) Собираем все PDB‑номера соседних остатков и объединяем с целевыми
    neighbor_resids = {res.resid for res in sel_neighbors.residues}
    all_pdb = {r for r, _ in target_pdb} | neighbor_resids

    # 7) Конвертируем обратно в 0‑based индексы ‑только той же цепи
    result = sorted(
        idx for idx, (r, ch) in seq_to_pdb.items()
        if (r in all_pdb and ch == chain_id)
    )
    return result

def get_align_alpha_phelix_idx(
    aligned_seq: list[int],
    idx: list[int],
) -> list[int]:
    
    '''
    Получает индексы альфа-спирали в выровненной последовательности.

    Параметры:
        aligned_seq (list[int]): выровненная последовательность, где -1 обозначает пропуски
        idx (list[int]): индексы интересующих позиций в оригинальной последовательности

    Возвращает:
        list[int]: индексы соответствующих позиций в выровненной последовательности
    '''

    idx_set = set(idx)
    result_map = {}
    original_pos = 0  # считает только те символы/коды, которые != gap_val

    for i, code in enumerate(aligned_seq):
        if code == -1:
            continue
        if original_pos in idx_set:
            result_map[original_pos] = i
        original_pos += 1

    # Собираем выходной список в том же порядке, что и входной,
    # но пропускаем те, которых нет в result_map
    return [result_map[i] for i in idx if i in result_map]

def get_original_alpha_phelix_idx(aligned_seq: list[int], idx: list[int]) -> list[int]:

    '''
    Получает индексы альфа-спирали в оригинальной последовательности из выровненной.

    Параметры:
        aligned_seq (list[int]): выровненная последовательность, где -1 обозначает пропуски
        idx (list[int]): индексы интересующих позиций в выровненной последовательности

    Возвращает:
        list[int]: индексы соответствующих позиций в оригинальной последовательности
    '''

    result = []
    new_pos = 0  # позиция в строке без '_'

    for i, elem in enumerate(aligned_seq):
        if elem == -1:
            continue
        if i in idx:
            result.append(new_pos)
        new_pos += 1

    return result

def active_side_transfer(seq1: str, seq2: str, idx: list[int]) -> list[int]:

    '''
    Переносит индексы активного сайта с одной последовательности на другую через выравнивание.

    Параметры:
        seq1 (str): исходная аминокислотная последовательность (с активным сайтом)
        seq2 (str): целевая аминокислотная последовательность
        idx (list[int]): индексы активного сайта в seq1

    Возвращает:
        list[int]: индексы активного сайта в seq2, полученные через выравнивание последовательностей
    '''

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

    # Например, [[ 0, 20,  1, -1, -1, -1],
    #            [ 0, 20,  1,  1,  1,  1]]
    seq1_code, seq2_code = align.get_codes(alignment)

    aligned_idx = get_align_alpha_phelix_idx(seq1_code, idx)
    original_idx = get_original_alpha_phelix_idx(seq2_code, aligned_idx)

    return original_idx