import os
import pytest
from fpgen.utils import fix_protein, get_active_site_residues
from esm.sdk.api import ESMProtein

TEST_PDB_PATH = 'tests/data/test_protein.pdb'

@pytest.fixture(scope='module')
def test_protein() -> ESMProtein:
    return ESMProtein.from_pdb(TEST_PDB_PATH)

def test_fix_protein_from_file(tmp_path):
    output_path = tmp_path / 'fixed.pdb'
    fixed_protein = fix_protein(input_filename=TEST_PDB_PATH, output_filename=str(output_path), platform='CUDA')

    assert isinstance(fixed_protein, ESMProtein)
    assert output_path.exists()
    with open(output_path) as f:
        lines = f.readlines()
        assert any('ATOM' in line for line in lines)

def test_fix_protein_from_esm(tmp_path, test_protein):
    output_path = tmp_path / 'fixed_from_esm.pdb'
    fixed_protein = fix_protein(protein=test_protein, output_filename=str(output_path), platform='CUDA')

    assert isinstance(fixed_protein, ESMProtein)
    assert output_path.exists()

def test_get_active_site_residues_from_file():
    residues = get_active_site_residues(
        target_residues=(57, 77),
        input_filename=TEST_PDB_PATH,
        radius=5.0
    )
    assert residues[:20] == [8, 12, 14, 16, 18, 27, 29, 35, 36, 37, 38, 39, 40, 41, 42, 44, 46, 47, 48, 53]
    assert isinstance(residues, list)
    assert all(isinstance(i, int) for i in residues)
    # Проверка на то что активный центр включен в индексы
    assert set(residues).intersection(set(range(56, 78)))

def test_get_active_site_residues_from_esm(test_protein):
    residues = get_active_site_residues(
        target_residues=(57, 77),
        protein=test_protein,
        radius=5.0
    )
    assert residues[:20] == [8, 12, 14, 16, 18, 27, 29, 35, 36, 37, 38, 39, 40, 41, 42, 44, 46, 47, 48, 53]
    assert isinstance(residues, list)
    assert all(isinstance(i, int) for i in residues)
