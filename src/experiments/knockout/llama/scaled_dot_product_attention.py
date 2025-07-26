import math
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    knockout_mask: Optional[Iterable[Tuple[int, int]]] = None,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    # print(attn_weight.shape, knockout_mask)
    # Apply attention knockout according to the knockout mask
    if knockout_mask is not None:
        for q, k in knockout_mask:
            attn_weight[:, :, q, k] = float("-inf")

    attn_weight = torch.softmax(attn_weight, dim=-1)
    # print(attn_weight[0,0])

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
