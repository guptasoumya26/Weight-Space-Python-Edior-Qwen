"""CPU-only, gradient-free edits of selected Qwen attention value biases."""

import math
from pathlib import Path
import torch

from . import MODEL_ID

LAYER_COUNT, HIDDEN, Q_HEADS, KV_HEADS, HEAD_DIM = 24, 896, 14, 2, 64
BIAS_SIZE = KV_HEADS * HEAD_DIM


def parse_layers(spec: str) -> list[int]:
    try:
        if ":" in spec:
            parts = spec.split(":")
            if len(parts) != 2:
                raise ValueError
            first, last = (int(x) for x in parts)
            result = list(range(first, last + 1))
        else:
            result = [int(x) for x in spec.split(",")]
    except ValueError as exc:
        raise ValueError("Use --layers 14:19 (inclusive) or comma-separated indices.") from exc
    if not result or len(set(result)) != len(result) or any(i < 0 or i >= LAYER_COUNT for i in result):
        raise ValueError("Layer indices must be unique and within 0..23, in ascending range order.")
    return result


def gqa_repeat_map(q_heads: int = Q_HEADS, kv_heads: int = KV_HEADS,
                   head_dim: int = HEAD_DIM) -> torch.Tensor:
    if q_heads <= 0 or kv_heads <= 0 or head_dim <= 0 or q_heads % kv_heads:
        raise ValueError("Query heads must be divisible by key/value heads.")
    repeat = q_heads // kv_heads
    result = torch.zeros((q_heads * head_dim, kv_heads * head_dim), dtype=torch.float32, device="cpu")
    for q in range(q_heads):
        kv = q // repeat
        for channel in range(head_dim):
            result[q * head_dim + channel, kv * head_dim + channel] = 1.0
    return result


def ridge_delta(output_weight: torch.Tensor, direction: torch.Tensor,
                ridge: float, max_update_norm: float) -> torch.Tensor:
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("--ridge must be finite and strictly positive.")
    if not math.isfinite(max_update_norm) or max_update_norm <= 0:
        raise ValueError("--max-update-norm must be finite and positive.")
    if tuple(output_weight.shape) != (HIDDEN, HIDDEN) or tuple(direction.shape) != (HIDDEN,):
        raise ValueError("Expected o_proj.weight [896,896] and direction [896].")
    repeat = gqa_repeat_map()
    a = output_weight.detach().to(device="cpu", dtype=torch.float32) @ repeat
    target = direction.detach().to(device="cpu", dtype=torch.float32)
    gram = a.T @ a + ridge * torch.eye(BIAS_SIZE, dtype=torch.float32)
    delta = torch.linalg.solve(gram, a.T @ target)
    norm = torch.linalg.vector_norm(delta)
    if not torch.isfinite(delta).all() or not torch.isfinite(norm):
        raise ValueError("Ridge solve returned a nonfinite update; increase --ridge.")
    if norm.item() > max_update_norm:
        delta = delta * (max_update_norm / norm)
    return delta


def validate_model(model, selected: list[int]) -> None:
    config = model.config
    expected = {"model_type": "qwen2", "num_hidden_layers": LAYER_COUNT,
                "hidden_size": HIDDEN, "num_attention_heads": Q_HEADS,
                "num_key_value_heads": KV_HEADS}
    for key, value in expected.items():
        if getattr(config, key, None) != value:
            raise ValueError(f"Unsupported architecture: expected {key}={value}, got {getattr(config, key, None)!r}.")
    actual_head_dim = getattr(config, "head_dim", HIDDEN // Q_HEADS)
    if actual_head_dim != HEAD_DIM:
        raise ValueError(f"Unsupported head dimension: expected {HEAD_DIM}, got {actual_head_dim}.")
    try:
        decoder_layers = model.model.layers
        embeddings = model.model.embed_tokens.weight
    except AttributeError as exc:
        raise ValueError("Expected a Qwen2ForCausalLM model with model.layers and embed_tokens.") from exc
    if len(decoder_layers) != LAYER_COUNT or embeddings.ndim != 2 or embeddings.shape[1] != HIDDEN:
        raise ValueError("Decoder layer count or embedding dimension does not match Qwen2.5-0.5B-Instruct.")
    for i in selected:
        bias = decoder_layers[i].self_attn.v_proj.bias
        weight = decoder_layers[i].self_attn.o_proj.weight
        if bias is None or tuple(bias.shape) != (BIAS_SIZE,) or tuple(weight.shape) != (HIDDEN, HIDDEN):
            raise ValueError(f"Layer {i} needs v_proj.bias [128] and o_proj.weight [896,896].")
        if bias.device.type != "cpu" or weight.device.type != "cpu":
            raise ValueError("The editor requires all selected tensors on CPU.")


def apply_edit(model, selected: list[int], direction: torch.Tensor, *,
               strength: float, ridge: float, max_update_norm: float,
               dry_run: bool = False) -> list[dict]:
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("--strength must be finite and nonnegative.")
    validate_model(model, selected)
    backups = {i: model.model.layers[i].self_attn.v_proj.bias.detach().clone() for i in selected}
    results = []
    try:
        with torch.no_grad():
            for i in selected:
                attn = model.model.layers[i].self_attn
                bias = attn.v_proj.bias
                delta = ridge_delta(attn.o_proj.weight, direction, ridge, max_update_norm)
                update = (strength * delta).to(dtype=bias.dtype)
                proposed = bias.detach() + update
                if not torch.isfinite(proposed).all():
                    raise ValueError(f"Layer {i} update is nonfinite; lower --strength.")
                if not dry_run:
                    bias.add_(update)
                    after = bias.detach()
                else:
                    after = proposed
                actual = after - backups[i]
                results.append({"layer": i, "tensor": f"model.layers.{i}.self_attn.v_proj.bias",
                                "addressed": BIAS_SIZE,
                                "nonzero_modified": int(torch.count_nonzero(actual).item()),
                                "before_norm": float(torch.linalg.vector_norm(backups[i].float()).item()),
                                "after_norm": float(torch.linalg.vector_norm(after.float()).item()),
                                "max_abs_delta": actual.abs().max().item(),
                                "actual_delta_norm": float(torch.linalg.vector_norm(actual.float()).item())})
    except Exception:
        with torch.no_grad():
            for i, original in backups.items():
                model.model.layers[i].self_attn.v_proj.bias.copy_(original)
        raise
    return results


def ensure_new_output(output: str, source: str, *, force: bool = False) -> Path:
    destination = Path(output).expanduser().resolve()
    source_path = Path(source).expanduser().resolve()
    if destination == source_path or (Path(source).expanduser().is_dir() and source_path in destination.parents):
        raise ValueError("Output must be outside the source model directory.")
    if destination.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {destination}. Choose a new path or pass --force.")
        if not destination.is_dir():
            raise ValueError("--force can replace only an output directory, not a file.")
        if any(destination.iterdir()):
            raise ValueError("For safety, --force accepts only an empty existing directory. Choose a new path.")
    if source == MODEL_ID and destination.name == MODEL_ID.split("/")[-1]:
        raise ValueError("Choose a distinct output directory for the edited checkpoint.")
    return destination
