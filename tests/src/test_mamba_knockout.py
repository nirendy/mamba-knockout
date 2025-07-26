import numpy as np
import pytest
import torch

from src.core.types import TSSM_A, TSSM_B, TSSM_C, KnockoutMode, TSSM_Bu, TSSMInput, TSSMState
from src.experiments.knockout.mamba.mamba1.helpers.knockout_scan import knockout_matrix, knockout_scan
from src.utils.type_checking import tensor_type_check

B_D = 1  # Batch dimension
INTRM_D = 1  # Intermediate dimension
SSM_D = 8  # SSM dimension
T = 8


@pytest.fixture
@tensor_type_check
def binary_test_data() -> tuple[TSSMState, TSSMInput, TSSM_A, TSSM_B, TSSM_Bu, TSSM_C]:
    """Fixture providing test data where inputs are powers of 2."""

    ssm_state = torch.zeros((B_D, INTRM_D, SSM_D))
    inputs = torch.Tensor([[[2**i for i in range(T)]]]).repeat(B_D, 1, 1)
    A = torch.ones((B_D, INTRM_D, T, SSM_D))
    B = torch.ones((B_D, INTRM_D, T, SSM_D)) / SSM_D  # Normalize inputs
    Bu = B * inputs.view(B_D, 1, T, 1)
    C = torch.ones((B_D, T, SSM_D))
    return ssm_state, inputs, A, B, Bu, C


@pytest.mark.parametrize(
    "knocked_out_inputs,affected_outputs,expected_values",
    [
        ([0, 1, 2], [4], [1.0, 3.0, 7.0, 15.0, 24.0, 63.0, 127.0, 255.0]),  #
        ([0, 1], [4], [1.0, 3.0, 7.0, 15.0, 28.0, 63.0, 127.0, 255.0]),  #
        ([], [], [1.0, 3.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0]),  # Test without knockouts
        ([], [4], [1.0, 3.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0]),  #
        ([0, 1, 2], [], [1.0, 3.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0]),  #
        ([0, 1, 2], [4, 5, 6], [1.0, 3.0, 7.0, 15.0, 24.0, 56.0, 120.0, 255.0]),  #
        ([0, 1], [4, 5, 6], [1.0, 3.0, 7.0, 15.0, 28.0, 60.0, 124.0, 255.0]),  #
    ],
)
def test_knockout_scan(
    binary_test_data,
    knocked_out_inputs,
    affected_outputs,
    expected_values,
):
    """Test knockout scan behavior with different knockout configurations."""
    ssm_state, inputs, discrete_A, B, deltaB_u, C = binary_test_data

    outputs = knockout_scan(
        seq_len=T,
        ssm_state=ssm_state,
        discrete_A=discrete_A,
        deltaB_u=deltaB_u,
        C=C,
        knocked_out_inputs=knocked_out_inputs,
        affected_outputs=affected_outputs,
        knockout_mode=KnockoutMode.ZERO_ATTENTION,
        dtype=torch.float32,
    )

    outputs_matrix = knockout_matrix(
        seq_len=T,
        discrete_A=discrete_A,
        discrete_B=B,
        C=C,
        u=inputs,
        knocked_out_inputs=knocked_out_inputs,
        affected_outputs=affected_outputs,
        dtype=torch.float32,
    )

    actual_values = torch.stack(outputs).view(-1).cpu().numpy()
    actual_values_matrix = outputs_matrix.view(-1).cpu().numpy()
    expected_values = np.array(expected_values)

    assert actual_values.tolist() == expected_values.tolist()
    assert actual_values_matrix.tolist() == expected_values.tolist()
