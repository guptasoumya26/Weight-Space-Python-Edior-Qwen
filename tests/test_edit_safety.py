"""Tiny synthetic components exercise editing and rollback without model I/O."""

from types import SimpleNamespace
import torch
import pytest

from qwen_weightedit.editor import apply_edit, ensure_new_output, parse_layers, validate_model


def fake_model():
    layers = []
    for _ in range(24):
        attention = SimpleNamespace(
            v_proj=SimpleNamespace(bias=torch.nn.Parameter(torch.zeros(128))),
            o_proj=SimpleNamespace(weight=torch.nn.Parameter(torch.eye(896))),
        )
        layers.append(SimpleNamespace(self_attn=attention))
    config = SimpleNamespace(model_type="qwen2", num_hidden_layers=24,
                             hidden_size=896, num_attention_heads=14,
                             num_key_value_heads=2, head_dim=64)
    return SimpleNamespace(config=config, model=SimpleNamespace(
        layers=layers, embed_tokens=SimpleNamespace(weight=torch.zeros(10, 896))))


def test_layer_parsing():
    assert parse_layers("14:19") == [14, 15, 16, 17, 18, 19]
    with pytest.raises(ValueError):
        parse_layers("23:24")
    with pytest.raises(ValueError):
        parse_layers("1,1")


def test_non_target_tensor_unchanged():
    model = fake_model()
    other = model.model.layers[15].self_attn.v_proj.bias.detach().clone()
    weight = model.model.layers[14].self_attn.o_proj.weight.detach().clone()
    direction = torch.ones(896) / (896 ** .5)
    result = apply_edit(model, [14], direction, strength=.25, ridge=.1, max_update_norm=.05)
    assert result[0]["addressed"] == 128
    assert result[0]["nonzero_modified"] > 0
    assert torch.equal(model.model.layers[15].self_attn.v_proj.bias, other)
    assert torch.equal(model.model.layers[14].self_attn.o_proj.weight, weight)


def test_dry_run_leaves_bias_unchanged():
    model = fake_model()
    direction = torch.ones(896)
    apply_edit(model, [14], direction, strength=.25, ridge=.1, max_update_norm=.05, dry_run=True)
    assert torch.count_nonzero(model.model.layers[14].self_attn.v_proj.bias) == 0


def test_rollback_on_second_layer_error(monkeypatch):
    import qwen_weightedit.editor as editor
    model = fake_model()
    original = editor.ridge_delta
    calls = 0

    def fails_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(editor, "ridge_delta", fails_second)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        apply_edit(model, [14, 15], torch.ones(896), strength=.25, ridge=.1, max_update_norm=.05)
    assert torch.count_nonzero(model.model.layers[14].self_attn.v_proj.bias) == 0


def test_shape_validation():
    model = fake_model()
    model.model.layers[14].self_attn.v_proj.bias = torch.nn.Parameter(torch.zeros(127))
    with pytest.raises(ValueError, match="v_proj.bias"):
        validate_model(model, [14])


def test_output_collision(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError):
        ensure_new_output(str(destination), "Qwen/Qwen2.5-0.5B-Instruct")
    with pytest.raises(ValueError, match="empty"):
        ensure_new_output(str(destination), "Qwen/Qwen2.5-0.5B-Instruct", force=True)
