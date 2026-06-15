import torch

from ai_services.explainability.attention_map import attention_rollout


def test_attention_rollout_returns_normalized_square_grid():
    # 2 layers, batch=1, 2 heads, seq_len = 1 (CLS) + 16 (4x4 patches) = 17
    seq_len = 17
    attentions = [torch.softmax(torch.randn(1, 2, seq_len, seq_len), dim=-1) for _ in range(2)]

    grid = attention_rollout(attentions)

    assert grid.shape == (4, 4)
    assert grid.min() >= 0.0
    assert grid.max() <= 1.0 + 1e-6


def test_attention_rollout_handles_single_layer():
    seq_len = 1 + 9  # 3x3 patch grid
    attentions = [torch.softmax(torch.randn(1, 4, seq_len, seq_len), dim=-1)]

    grid = attention_rollout(attentions)

    assert grid.shape == (3, 3)
