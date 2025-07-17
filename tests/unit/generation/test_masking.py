import torch
import pytest
from unittest.mock import MagicMock
from esm.sdk.api import ESMProtein
from fpgen.generation.masking import get_masked_protein

@pytest.fixture
def example_protein():
    # Простая белковая последовательность для теста
    return ESMProtein(sequence='ACDEFGHIKLMNPQRSTVWY')

@pytest.fixture
def mock_model():
    mock = MagicMock()

    def encode(protein):
        seq_len = len(protein.sequence)
        fake_encoded = MagicMock()
        fake_encoded.sequence = torch.zeros(seq_len + 2, dtype=torch.long)  # [CLS] + seq + [EOS]
        fake_encoded.structure = torch.ones(seq_len + 2, dtype=torch.long) * 4096
        return fake_encoded

    mock.encode.side_effect = encode
    return mock

def test_get_masked_protein_length(example_protein, mock_model):
    unmasked = [1, 3, 5]
    masked_protein = get_masked_protein(example_protein, unmasked, mock_model)

    assert isinstance(masked_protein, MagicMock)  # возвращается результат mock.encode
    assert masked_protein.structure.shape == masked_protein.sequence.shape
    assert len(masked_protein.sequence) == len(example_protein.sequence) + 2

def test_structure_masking_behavior(example_protein, mock_model):
    unmasked = [0, 2, 4]
    result = get_masked_protein(example_protein, unmasked, mock_model)

    print(result.sequence)
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
