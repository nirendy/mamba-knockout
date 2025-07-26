import numpy as np
import pytest
import torch

from src.experiments.knockout.llama.sdpa_attention import sdpa_attention_forward
from src.utils.type_checking import tensor_type_check

B_D = 1  # Batch dimension
INTRM_D = 8  # Intermediate dimension
T = 8

TQ = torch.Tensor
TK = torch.Tensor
TV = torch.Tensor


@pytest.fixture
@tensor_type_check
def binary_test_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture providing test data where inputs are powers of 2."""

    q: torch.Tensor = torch.ones((B_D, T, INTRM_D))
    k: torch.Tensor = torch.ones((B_D, T, INTRM_D))
    v: torch.Tensor = 2 ** torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(B_D, T, 1)

    return q, k, v


@pytest.mark.parametrize(
    "knockout_mask,expected_values",
    [
        ([(0, 4), (1, 4), (2, 4)], [1.0, 3.0, 7.0, 15.0, 24.0, 63.0, 127.0, 255.0]),  #
        ([(0, 1)], [1.0, 2.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0]),  #
        ([(1, 2)], [1.0, 3.0, 5.0, 15.0, 31.0, 63.0, 127.0, 255.0]),  #
        ([(0, 4), (1, 4)], [1.0, 3.0, 7.0, 15.0, 28.0, 63.0, 127.0, 255.0]),  #
        ([], [1.0, 3.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0]),  # Test without knockouts
        (
            [
                (0, 4),
                (1, 4),
                (2, 4),
                (0, 5),
                (1, 5),
                (2, 5),
                (0, 6),
                (1, 6),
                (2, 6),
            ],
            [1.0, 3.0, 7.0, 15.0, 24.0, 56.0, 120.0, 255.0],
        ),  #
        ([(0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6)], [1.0, 3.0, 7.0, 15.0, 28.0, 60.0, 124.0, 255.0]),  #
    ],
)
def test_knockout(
    binary_test_data,
    knockout_mask,
    expected_values,
):
    """Test knockout scan behavior with different knockout configurations."""
    q, k, v = binary_test_data

    actual_outputs, _ = sdpa_attention_forward(
        None,
        query=q,
        key=k,
        value=v,
        attention_mask=None,
        dropout=0.0,
        knockout_mask=knockout_mask,
    )

    actual_values = actual_outputs.view(-1).cpu().numpy()

    values = v.view(-1).cpu().numpy()
    calculated_expected_outputs = values.cumsum()

    expected_values = np.array(expected_values)
    T = len(expected_values)
    denom = np.arange(1, T + 1)
    for t in knockout_mask:
        denom[t[1]] -= 1
        calculated_expected_outputs[t[1]] -= values[t[0]]
    expected_values = expected_values / denom
    calculated_expected_outputs = calculated_expected_outputs / denom

    print("actual_values", actual_values)
    print("expected_values", expected_values)
    print("full_outputs", calculated_expected_outputs)

    assert np.isclose(actual_values, calculated_expected_outputs, rtol=1e-5).all()
    assert np.isclose(actual_values, expected_values, rtol=1e-5).all()
