from typing import Iterable, List, Optional

import torch
from jaxtyping import Float
from torch import Tensor, matmul, zeros_like

from src.core.types import TSSM_A, TSSM_B, TSSM_C, KnockoutMode, TSSM_Bu, TSSMInput, TSSMState
from src.utils.type_checking import tensor_type_check


@tensor_type_check
def knockout_scan(
    seq_len: int,
    ssm_state: TSSMState,
    discrete_A: TSSM_A,
    deltaB_u: TSSM_Bu,
    C: TSSM_C,
    knocked_out_inputs: Iterable[int],
    affected_outputs: Iterable[int],
    knockout_mode: KnockoutMode,
    dtype: torch.dtype,
    knockout_feature_mask: Optional[torch.FloatTensor | torch.Tensor] = None,
) -> List[Float[Tensor, "batch hidden_size"]]:  # noqa: F722
    knockout_state: TSSMState = zeros_like(ssm_state)
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
        if i not in knocked_out_inputs:
            knockout_state = discrete_A[:, :, i, :] * knockout_state + deltaB_u[:, :, i, :]
        else:
            if knockout_mode in {KnockoutMode.ZERO_ATTENTION}:
                knockout_state = discrete_A[:, :, i, :] * knockout_state
            elif knockout_mode in {KnockoutMode.ZERO_DELTA}:
                knockout_state = knockout_state

            if knockout_feature_mask is not None:
                knockout_state = (
                    knockout_state
                    + knockout_feature_mask.view(1, -1, 1).float().expand_as(deltaB_u[:, :, i, :])
                    * deltaB_u[:, :, i, :]
                )

        scan_output = torch.einsum(
            "bij,bj->bi",
            (knockout_state if (i in affected_outputs) else ssm_state).to(dtype),
            C[:, i, :],
        )
        scan_outputs.append(scan_output)

    return scan_outputs


def materialize_ssm_transition(A: torch.Tensor) -> torch.Tensor:
    batch = A.shape[0]
    D = A.shape[1]
    T = A.shape[2]
    N = A.shape[3]
    A = A.transpose(-1, -2).repeat(1, 1, 1, T).reshape(batch, D, N, T, T).transpose(-1, -2)
    A = torch.tril(A) + torch.triu(torch.ones_like(A), 1)
    A_cumprod = torch.cumprod(A, dim=-2)

    transition_mat = A_cumprod.transpose(-2, -3)

    return transition_mat


def materialize_ssm_attention(
    A: Tensor, B: Tensor, C: Tensor, return_transition: bool
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    transition_mat = materialize_ssm_transition(A)

    AB = transition_mat * B.unsqueeze(-1)

    out = torch.einsum("btn, bdtnq -> bdtq", C, AB)

    if return_transition:
        return out, transition_mat

    return torch.tril(out)


@tensor_type_check
def compute_attn_matrix_fn(
    dA: TSSM_A,
    dB: TSSM_B,
    C: TSSM_C,
    L: int,
    x_shape: tuple,
    dtype: torch.dtype = torch.float16,
) -> Float[Tensor, "batch hidden_size seq_len seq_len"]:  # noqa: F722
    # dA = torch.exp(torch.einsum("bdl,dn->bldn", dt, A))
    # dB = torch.einsum("bdl,bnl->bldn", dt, B.squeeze(1))
    AttnMatrixOverCLS = (
        torch.zeros((x_shape[0], x_shape[1], x_shape[2], x_shape[2])).to(dtype).to(dA.device)
    )  # BHLL: L vectors per batch and channel
    for r in range(L):
        for c in range(r + 1):
            curr_C = C[:, r, :]
            currA = torch.ones((dA.shape[0], dA.shape[1], dA.shape[3]), dtype=dtype).to(dA.device)
            if c < r:
                for i in range(r - c):
                    currA = currA * dA[:, :, r - i, :]
            currB = dB[:, :, c, :]
            AttnMatrixOverCLS[:, :, r, c] = torch.sum(curr_C * currA * currB, axis=-1)  # type: ignore
    return AttnMatrixOverCLS


@tensor_type_check
def knockout_matrix(
    seq_len: int,
    discrete_A: TSSM_A,
    discrete_B: TSSM_B,
    u: TSSMInput,
    C: TSSM_C,
    knocked_out_inputs: Iterable[int],
    affected_outputs: Iterable[int],
    dtype,
) -> Float[Tensor, "batch hidden_size seq_len"]:  # noqa: F722
    attn = compute_attn_matrix_fn(discrete_A, discrete_B, C, seq_len, u.shape, dtype)
    for i in affected_outputs:
        for j in knocked_out_inputs:
            attn[:, :, i, j] = 0
    outputs = torch.einsum("bdtx,bdx->bdt", attn, u)
    return outputs


def ignore_context_knockout_scan(
    seq_len: int,
    ssm_state: Tensor,
    discrete_A: Tensor,
    deltaB_u: Tensor,
    C: Tensor,
    knockout_start_idx: int,
    knockout_end_idx: int,
    dtype,
) -> List[Tensor]:
    knockout_state = zeros_like(ssm_state)
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]  # [batch, intermediade_size, ssm_state]
        # TODO: Test this to see if it works, (prime numbers)
        if (i >= knockout_start_idx) and (i < knockout_end_idx):
            knockout_state = (
                discrete_A[:, :, i, :] * knockout_state + deltaB_u[:, :, i, :]
            )  # [batch, intermediade_size, ssm_state]
            scan_output = matmul(knockout_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
        else:
            scan_output = matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
        scan_outputs.append(scan_output[:, :, 0])

    return scan_outputs
