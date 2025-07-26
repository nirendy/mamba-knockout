from typing import Iterable, Optional

import torch
from torch import nn
from transformers.cache_utils import MambaCache

from src.core.types import KnockoutMode
from src.experiments.knockout.mamba.mamba1.helpers.knockout_scan import knockout_matrix, knockout_scan


def rms_forward(hidden_states, variance_epsilon=1e-6):
    """
    Calculates simple RMSNorm with no learnable weights. `MambaRMSNorm` will
    leverage this in order to multiply the final result with the RMSNorm weight

    Args:
        hidden_states (`torch.Tensor`):
            Hidden states to normalize
        variance_epsilon (`float`):
            The eps value to add in the square root scaling factor
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def slow_forward_for_ssm_materializing_knockout_falcon(
    self,
    input_states,
    cache_params: Optional[MambaCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    knockout_indices: Optional[Iterable[int]] = None,
    affected_outputs: Optional[Iterable[int]] = None,
    knockout_mode: Optional[KnockoutMode] = None,
    knockout_feature_mask: Optional[torch.Tensor | torch.BoolTensor | torch.FloatTensor] = None,
    with_materialized_attention_matrix: Optional[bool] = False,
):
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(1, 2)  # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 2. Convolution sequence transformation
    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx].clone()
        ssm_state = ssm_state.to(hidden_states.device)
        # use `cache_position.shape[0]` to check whether we are in prefill
        # stage, it's equivalent to check `cache_position[0] == 0`, which
        # breaks dynamo fullgraph constraints
        if cache_position is not None and cache_position.shape[0] == self.conv_kernel_size:
            conv_state = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))

            cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])  # [batch, intermediate_size, seq_len]
        else:
            conv_state = cache_params.update_conv_state(
                self.layer_idx,
                hidden_states,
                cache_position,  # type: ignore
            )
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)  # [batch, intermediate_size, 1] : decoding
    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size), device=hidden_states.device, dtype=dtype
        )
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])  # [batch, intermediate_size, seq_len]

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
    )

    B = rms_forward(B, variance_epsilon=self.rms_eps)
    C = rms_forward(C, variance_epsilon=self.rms_eps)
    time_step = rms_forward(time_step, variance_epsilon=self.rms_eps)

    discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(
        1, 2
    )  # [batch, intermediate_size, seq_len]

    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(self.A_log.float())  # [intermediate_size, ssm_state_size]
    discrete_A = torch.exp(
        A[None, :, None, :] * discrete_time_step[:, :, :, None]
    )  # [batch, intermediate_size, seq_len, ssm_state_size]
    discrete_B = (
        discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    )  # [batch, intermediate_size, seq_len, ssm_state_size]
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    # 3. SSM Interference
    if with_materialized_attention_matrix:
        u = hidden_states[:, :, :, None].float()
        scan_output = knockout_matrix(seq_len, discrete_A, discrete_B, u, C, knockout_indices, affected_outputs, dtype)  # type: ignore
    else:
        scan_outputs = knockout_scan(
            seq_len,
            ssm_state,
            discrete_A,
            deltaB_u,
            C,
            knockout_indices,  # type: ignore
            affected_outputs,  # type: ignore
            knockout_mode,  # type: ignore
            dtype,
            knockout_feature_mask,
        )
        scan_output = torch.stack(scan_outputs, dim=-1)  # [batch, seq_len, intermediade_size]
    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)

    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
    return contextualized_states
