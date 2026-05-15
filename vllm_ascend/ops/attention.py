# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Ascend-specific attention operations for vLLM.
# This module provides NPU-optimized attention kernels that replace
# the default CUDA-based implementations in upstream vLLM.

from typing import List, Optional, Tuple

import torch

try:
    import torch_npu  # type: ignore
except ImportError:
    raise ImportError(
        "torch_npu is required for Ascend NPU attention operations. "
        "Please install the appropriate torch_npu package."
    )


def paged_attention_npu(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
) -> torch.Tensor:
    """NPU-optimized paged attention for decode phase.

    Args:
        query: Query tensor of shape [num_seqs, num_heads, head_size].
        key_cache: Paged key cache of shape [num_blocks, num_kv_heads, block_size, head_size].
        value_cache: Paged value cache of shape [num_blocks, num_kv_heads, block_size, head_size].
        block_tables: Block table of shape [num_seqs, max_num_blocks_per_seq].
        context_lens: Context lengths of shape [num_seqs].
        scale: Attention scale factor (typically 1 / sqrt(head_size)).
        alibi_slopes: Optional ALiBi slopes of shape [num_heads].
        kv_cache_dtype: Data type for kv cache ("auto", "fp8_e4m3", etc.).

    Returns:
        Output tensor of shape [num_seqs, num_heads, head_size].
    """
    num_seqs, num_heads, head_size = query.shape
    block_size = key_cache.shape[2]
    max_context_len = int(context_lens.max().item())

    output = torch.empty_like(query)

    # Use torch_npu's flash attention decode kernel when available
    # Falls back to a manual gather-and-compute approach otherwise
    try:
        output = torch_npu.npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_tables,
            context_lens=context_lens,
            scale_value=scale,
        )
    except AttributeError:
        # Fallback: manual implementation using standard NPU ops
        output = _paged_attention_fallback(
            query, key_cache, value_cache,
            block_tables, context_lens,
            scale, max_context_len, block_size,
        )

    return output


def _paged_attention_fallback(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    max_context_len: int,
    block_size: int,
) -> torch.Tensor:
    """Fallback paged attention implementation using standard PyTorch ops.

    This is used when the NPU-native paged attention kernel is not available.
    Performance is suboptimal compared to the native kernel.
    """
    num_seqs, num_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    output = torch.zeros_like(query)

    for seq_idx in range(num_seqs):
        ctx_len = int(context_lens[seq_idx].item())
        num_blocks = (ctx_len + block_size - 1) // block_size

        # Gather key and value blocks for this sequence
        gathered_keys: List[torch.Tensor] = []
        gathered_vals: List[torch.Tensor] = []

        for blk_idx in range(num_blocks):
            physical_block = int(block_tables[seq_idx, blk_idx].item())
            gathered_keys.append(key_cache[physical_block])    # [num_kv_heads, block_size, head_size]
            gathered_vals.append(value_cache[physical_block])  # [num_kv_heads, block_size, head_size]

        # Shape: [num_kv_heads, ctx_len, head_size]
        keys = torch.cat(gathered_keys, dim=1)[:, :ctx_len, :]
        vals = torch.cat(gathered_vals, dim=1)[:, :ctx_len, :]

        # Expand kv heads to match query heads (for MQA/GQA)
        if num_kv_heads != num_heads:
            assert num_heads % num_kv_heads == 0
            repeat_factor = num_heads // num_kv_heads
            keys = keys.repeat_interleave(repeat_factor, dim=0)
            vals = vals.repeat_interleave(repeat_factor, dim=0)

        # q: [num_heads, 1, head_size], k: [num_heads, ctx_len, head_size]
        q = query[seq_idx].unsqueeze(1)
        attn_weights = torch.bmm(q, keys.transpose(1, 2)) * scale  # [num_heads, 1, ctx_len]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output[seq_idx] = torch.bmm(attn_weights, vals).squeeze(1)  # [num_heads, head_size]

    return output
