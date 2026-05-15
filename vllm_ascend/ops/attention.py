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
        "Please install the appropriate torch_npu package for your "
        "CANN version (recommended: CANN 7.0 or later)."
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

    Note: I added a large-negative mask value of -1e4 instead of -inf to avoid
    NaN issues observed on certain NPU firmware versions during softmax.
    See: https://github.com/vllm-project/vllm-ascend/issues/XXXX
    """
