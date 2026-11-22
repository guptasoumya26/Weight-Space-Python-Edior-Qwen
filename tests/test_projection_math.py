"""Synthetic tests only. No test loads or downloads a model."""

import torch
import pytest

from qwen_weightedit.editor import gqa_repeat_map, ridge_delta


def test_gqa_repeats_whole_head_blocks():
    mapping = gqa_repeat_map(q_heads=4, kv_heads=2, head_dim=3)
    value = torch.tensor([1., 2., 3., 4., 5., 6.])
    assert (mapping @ value).tolist() == [1., 2., 3., 1., 2., 3., 4., 5., 6., 4., 5., 6.]


def test_ridge_solve_is_finite_and_capped():
    weight = torch.eye(896)
    target = torch.ones(896) / (896 ** .5)
    delta = ridge_delta(weight, target, ridge=.1, max_update_norm=.05)
    assert delta.shape == (128,)
    assert torch.isfinite(delta).all()
    assert torch.linalg.vector_norm(delta).item() <= .050001


def test_projection_rejects_bad_shapes_and_regularization():
    with pytest.raises(ValueError, match="Expected o_proj"):
        ridge_delta(torch.eye(16), torch.ones(896), .1, .05)
    with pytest.raises(ValueError, match="ridge"):
        ridge_delta(torch.eye(896), torch.ones(896), 0, .05)
