from typing import Iterable, Optional

import torch
from torch import nn
from transformers.cache_utils import MambaCache

from src.core.types import KnockoutMode
from src.experiments.knockout.mamba.mamba1.helpers.knockout_scan import knockout_matrix, knockout_scan


# fmt: off
def slow_forward_for_ssm_materializing_knockout(
        module,
        input_states,
        cache_params: Optional[MambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.LongTensor] = None,
        knockout_indices: Optional[Iterable[int]] = None, affected_outputs: Optional[Iterable[int]] = None,
        knockout_mode: Optional[KnockoutMode] = None,
        knockout_feature_mask: Optional[torch.Tensor | torch.BoolTensor | torch.FloatTensor] = None,
        with_materialized_attention_matrix: Optional[bool] = False
):
    """
    The implementation of MambaMixer's forward pass, 
    updated to return the calculated SSM parameters (A, B, C) for analysis.
    """
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    # 1. Gated MLP's linear projection
    projected_states = module.in_proj(input_states).transpose(1, 2)  # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 2. Convolution sequence transformation
    if cache_params is not None:
        assert cache_position is not None
        ssm_state = cache_params.ssm_states[module.layer_idx].clone()
        ssm_state = ssm_state.to(hidden_states.device)
        # use `cache_position.shape[0]` to check whether we are in prefill
        # stage, it's equivalent to check `cache_position[0] == 0`, which
        # breaks dynamo fullgraph constraints
        if cache_position.shape[0] == module.conv_kernel_size:
            conv_state = nn.functional.pad(
                hidden_states,
                (module.conv_kernel_size - hidden_states.shape[-1], 0)
            )

            cache_params.update_conv_state(module.layer_idx, conv_state, cache_position)
            hidden_states = module.act(
                module.conv1d(hidden_states)[..., :seq_len])  # [batch, intermediate_size, seq_len]
        else:
            conv_state = cache_params.update_conv_state(module.layer_idx, hidden_states, cache_position)
            hidden_states = torch.sum(conv_state * module.conv1d.weight[:, 0, :], dim=-1)
            if module.use_conv_bias:
                hidden_states += module.conv1d.bias
            hidden_states = module.act(hidden_states).to(dtype).unsqueeze(
                -1)  # [batch, intermediate_size, 1] : decoding
    else:
        ssm_state = torch.zeros(
            (batch_size, module.intermediate_size, module.ssm_state_size),
            device=hidden_states.device, dtype=dtype
        )
        # knockout_input_state = torch.zeros(
        #     (batch_size, module.intermediate_size, module.ssm_state_size),
        #     device=hidden_states.device, dtype=dtype
        # )
        final_state = torch.zeros(  # noqa: F841
            (batch_size, module.intermediate_size, module.ssm_state_size),
            device=hidden_states.device, dtype=dtype
        )
        hidden_states = module.act(module.conv1d(hidden_states)[..., :seq_len])  # [batch, intermediate_size, seq_len]

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = module.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters, [module.time_step_rank, module.ssm_state_size, module.ssm_state_size], dim=-1
    )
    discrete_time_step = module.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1,
                                                                              2)  # [batch, intermediate_size, seq_len]

    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(module.A_log.float())  # [intermediate_size, ssm_state_size]
    discrete_A = torch.exp(
        A[None, :, None, :] * discrete_time_step[:, :, :, None])  # [batch, intermediate_size, seq_len, ssm_state_size]
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :,
                                                     :].float()  # [batch, intermediate_size, seq_len, ssm_state_size]
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    # 3.c perform the recurrence y ← SSM(A, B, C)(x)
    # Here is the call to the knockout_scan functions

    if with_materialized_attention_matrix:
        u = hidden_states[:, :, :, None].float()
        scan_output = knockout_matrix(seq_len, discrete_A, discrete_B, u, C, knockout_indices, affected_outputs, dtype)  # type: ignore
    else:
        scan_outputs = knockout_scan(
            seq_len, ssm_state, discrete_A, deltaB_u, C, 
            knockout_indices, affected_outputs, knockout_mode, dtype, knockout_feature_mask)  # type: ignore
        scan_output = torch.stack(scan_outputs, dim=-1)  # [batch, seq_len, intermediade_size]
    scan_output = scan_output + (hidden_states * module.D[None, :, None])
    scan_output = (scan_output * module.act(gate))

    if cache_params is not None:
        cache_params.ssm_states[module.layer_idx].copy_(ssm_state)

    # 4. Final linear projection
    contextualized_states = module.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
    return contextualized_states
# fmt: on
