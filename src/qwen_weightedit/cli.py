"""Command line entry points. Imports never download or load a model."""

import argparse
import json
import math
from pathlib import Path
import sys

from . import __version__, MODEL_ID
from .concepts import embedding_direction
from .editor import apply_edit, ensure_new_output, parse_layers, validate_model
from .inference import SYSTEM_PROMPT, generate_batch, load_model, load_tokenizer


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m qwen_weightedit")
    commands = p.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect the supported model architecture")
    edit = commands.add_parser("edit", help="Apply an experimental bias edit and save a new checkpoint")
    compare = commands.add_parser("compare", help="Generate matched baseline and edited responses")
    for command in (inspect, edit, compare):
        command.add_argument("--model", default=MODEL_ID)
        command.add_argument("--local-files-only", action="store_true")
    edit.add_argument("--concept", default="debugging")
    edit.add_argument("--layers", default="14:19")
    edit.add_argument("--strength", type=float, default=0.25)
    edit.add_argument("--ridge", type=float, default=0.1)
    edit.add_argument("--max-update-norm", type=float, default=0.05)
    edit.add_argument("--output", required=True)
    edit.add_argument("--force", action="store_true", help="Permit an existing empty output directory")
    edit.add_argument("--dry-run", action="store_true", help="Compute previews without changing or saving a model")
    edit.add_argument("--seed", type=int, default=42)
    compare.add_argument("--edited", required=True)
    compare.add_argument("--prompts", required=True)
    compare.add_argument("--max-new-tokens", type=int, default=80)
    compare.add_argument("--output", required=True)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--force", action="store_true")
    return p


def check_model_id(value: str, *, allow_edited: bool = False) -> None:
    if value == MODEL_ID:
        return
    path = Path(value).expanduser()
    if path.is_dir() and (path / "config.json").is_file():
        return
    kind = "a local Qwen checkpoint" if allow_edited else "a local Qwen2.5-0.5B-Instruct checkpoint"
    raise ValueError(f"Use {MODEL_ID} or {kind} directory containing config.json.")


def run_inspect(args) -> None:
    model = load_model(args.model, local_files_only=args.local_files_only)
    validate_model(model, list(range(24)))
    print(f"Supported architecture: {args.model}; 24 layers, hidden 896, 14 Q / 2 KV heads, v_proj.bias [128].")


def run_edit(args) -> None:
    check_model_id(args.model)
    selected = parse_layers(args.layers)
    output = ensure_new_output(args.output, args.model, force=args.force)
    if not math.isfinite(args.strength) or args.strength < 0:
        raise ValueError("--strength must be finite and nonnegative.")
    if not math.isfinite(args.ridge) or args.ridge <= 0:
        raise ValueError("--ridge must be finite and positive.")
    if not math.isfinite(args.max_update_norm) or args.max_update_norm <= 0:
        raise ValueError("--max-update-norm must be finite and positive.")
    import torch
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.model, local_files_only=args.local_files_only)
    model = load_model(args.model, local_files_only=args.local_files_only)
    validate_model(model, selected)
    direction = embedding_direction(tokenizer, model.model.embed_tokens.weight, args.concept)
    results = apply_edit(model, selected, direction, strength=args.strength,
                         ridge=args.ridge, max_update_norm=args.max_update_norm,
                         dry_run=args.dry_run)
    addressed = sum(row["addressed"] for row in results)
    changed = sum(row["nonzero_modified"] for row in results)
    change_label = "would_change" if args.dry_run else "changed"
    for row in results:
        print(f"layer {row['layer']:02d} {row['tensor']}: addressed={row['addressed']} "
              f"{change_label}={row['nonzero_modified']} l2_delta={row['actual_delta_norm']:.6g} "
              f"max_abs_delta={row['max_abs_delta']:.6g}")
    print(f"Total: addressed={addressed} {change_label}={changed} dry_run={args.dry_run}")
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    metadata = {
        "source_model": args.model, "model_family": MODEL_ID,
        "concept": args.concept, "layers": selected, "strength": args.strength,
        "ridge": args.ridge, "max_update_norm": args.max_update_norm,
        "algorithm": "embedding-contrast-gqa-ridge-v-proj-bias-v1",
        "code_version": __version__, "device": "cpu", "dtype": "float32",
        "addressed_parameters": addressed, "nonzero_modified_parameters": changed,
        "bias_summaries": results, "behavioral_comparison_executed": False,
        "limitations": "Embedding contrast and simplified linear attention map; no guaranteed semantic steering.",
    }
    (output / "edit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved edited checkpoint and tokenizer to {output}.")


def run_compare(args) -> None:
    check_model_id(args.model)
    if not Path(args.edited).expanduser().is_dir():
        raise ValueError("--edited must be a local checkpoint directory produced by edit.")
    check_model_id(args.edited, allow_edited=True)
    if not (Path(args.edited).expanduser() / "edit_metadata.json").is_file():
        raise ValueError("Edited checkpoint is missing edit_metadata.json; run edit first.")
    if not 1 <= args.max_new_tokens <= 80:
        raise ValueError("--max-new-tokens must be between 1 and 80.")
    path = Path(args.output).expanduser().resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Comparison --output must be a .json file.")
    edited_dir = Path(args.edited).expanduser().resolve()
    if edited_dir in path.parents or (Path(args.model).expanduser().is_dir()
                                      and Path(args.model).expanduser().resolve() in path.parents):
        raise ValueError("Comparison output must be outside both model directories.")
    if path.exists() and not args.force:
        raise FileExistsError(f"Comparison output exists: {path}; use a new path or --force.")
    baseline_template = load_tokenizer(args.model, local_files_only=args.local_files_only).chat_template
    edited_template = load_tokenizer(args.edited, local_files_only=True).chat_template
    if baseline_template != edited_template:
        raise ValueError("Tokenizer chat templates differ; comparison requires identical templates.")
    source = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if not isinstance(source, list) or not source or any(
        not isinstance(x, dict) or not isinstance(x.get("id"), str)
        or not isinstance(x.get("prompt"), str) or not x["prompt"].strip()
        for x in source
    ):
        raise ValueError("Prompts file must be a nonempty JSON list of {id, prompt} objects.")
    if len(source) > 20:
        raise ValueError("Limit comparison to at most 20 prompts.")
    baseline = generate_batch(args.model, source, local_files_only=args.local_files_only,
                              max_new_tokens=args.max_new_tokens, seed=args.seed)
    edited = generate_batch(args.edited, source, local_files_only=True,
                            max_new_tokens=args.max_new_tokens, seed=args.seed)
    result = {"executed": True, "source_model": args.model, "edited_model": str(Path(args.edited).resolve()),
              "settings": {"system_prompt": SYSTEM_PROMPT, "do_sample": False,
                           "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                           "chat_template": "model tokenizer apply_chat_template"},
              "responses": [{"id": x["id"], "prompt": x["prompt"], "baseline": b, "edited": e}
                            for x, b, e in zip(source, baseline, edited)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(source)} actual matched response pairs to {path}.")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        check_model_id(args.model)
        {"inspect": run_inspect, "edit": run_edit, "compare": run_compare}[args.command](args)
    except (ValueError, FileNotFoundError, FileExistsError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
