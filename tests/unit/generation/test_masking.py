import torch
import pytest
from unittest.mock import MagicMock
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from fpgen.generation.masking import get_masked_protein
from esm.models.esm3 import ESM3

@pytest.fixture
def example_protein():
    # Белок для теста
    return ESMProtein.from_protein_chain(ProteinChain.from_rcsb('1qy3', chain_id='A'))

@pytest.fixture
def model():
    model = ESM3.from_pretrained('esm3-open').to('cuda')
    return model

def test_get_masked_protein_length(example_protein, model):
    unmasked = [1, 3, 5, 10, 20]
    masked_protein = get_masked_protein(example_protein, unmasked, model)

    assert masked_protein.structure.shape == masked_protein.sequence.shape
    assert len(masked_protein.sequence) == len(example_protein.sequence) + 2

def test_structure_masking_behavior(example_protein, model):
    unmasked = [1, 3, 5, 10, 20]
    result = get_masked_protein(example_protein, unmasked, model)

    # Проверим, что structure[1 + i] скопирован из protein_tokens.structure
    for i in range(len(example_protein.sequence)):
        token_value = result.structure[i + 1].item()
        if i in unmasked:
            assert token_value != 4096  # не должен быть замаскирован
        else:
            assert token_value == 4096  # должен быть замаскирован

    # Проверим специальные токены
    assert result.structure[0].item() == 4098
    assert result.structure[-1].item() == 4097
