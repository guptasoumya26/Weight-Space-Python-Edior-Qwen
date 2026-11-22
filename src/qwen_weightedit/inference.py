"""Sequential, matched CPU generation for the original and edited checkpoints."""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .editor import validate_model

SYSTEM_PROMPT = "You are a helpful assistant. Answer briefly."


def load_model(path: str, *, local_files_only: bool):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, device_map=None,
        local_files_only=local_files_only, trust_remote_code=False,
    )
    model.to("cpu")
    model.eval()
    return model


def load_tokenizer(path: str, *, local_files_only: bool):
    return AutoTokenizer.from_pretrained(path, local_files_only=local_files_only,
                                         trust_remote_code=False)


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=True,
                                            return_tensors="pt")
    if encoded.shape[1] > 2048:
        raise ValueError("Prompt exceeds the 2048-token comparison limit.")
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode():
        output = model.generate(encoded.to("cpu"), max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=pad,
                                eos_token_id=eos)
    return tokenizer.decode(output[0, encoded.shape[1]:], skip_special_tokens=True).strip()


def generate_batch(path: str, prompts: list[dict], *, local_files_only: bool,
                   max_new_tokens: int, seed: int) -> list[str]:
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(path, local_files_only=local_files_only)
    model = load_model(path, local_files_only=local_files_only)
    try:
        validate_model(model, list(range(24)))
        return [generate_one(model, tokenizer, item["prompt"], max_new_tokens)
                for item in prompts]
    finally:
        del model, tokenizer
        gc.collect()
