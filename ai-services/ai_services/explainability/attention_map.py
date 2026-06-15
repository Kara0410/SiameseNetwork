"""Attention-rollout explainability for transformer vision encoders (ViT, CLIP).

Implements the attention-rollout method (Abnar & Zuidema, 2020): per-layer attention
matrices are recursively multiplied (with an identity term for the residual
connection) to approximate how strongly each output token attends back to each input
patch. The CLS token's row of the rolled-out matrix is reshaped into the patch grid to
produce a heatmap.
"""

import numpy as np
import torch


def attention_rollout(attentions: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> np.ndarray:
    """Turn a stack of per-layer attention tensors into a square `[0, 1]` heatmap.

    Args:
        attentions: one `(batch, heads, seq_len, seq_len)` tensor per transformer
            layer, as returned by `output_attentions=True`. `seq_len` must be
            `1 + grid_size**2` (a leading CLS token followed by a square patch grid).

    Returns:
        A `(grid_size, grid_size)` numpy array normalized to `[0, 1]`.
    """
    seq_len = attentions[0].shape[-1]
    rolled = torch.eye(seq_len, device=attentions[0].device)

    for layer_attention in attentions:
        # Average over attention heads for the first (only) item in the batch.
        attn = layer_attention[0].mean(dim=0)
        attn = attn + torch.eye(seq_len, device=attn.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rolled = attn @ rolled

    cls_to_patches = rolled[0, 1:]
    grid_size = int(cls_to_patches.numel() ** 0.5)
    grid = cls_to_patches[: grid_size * grid_size].reshape(grid_size, grid_size)

    heatmap = grid.cpu().numpy().astype(np.float32)
    heatmap -= heatmap.min()
    max_value = heatmap.max()
    if max_value > 0:
        heatmap /= max_value
    return heatmap
