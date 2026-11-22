"""A documented embedding contrast heuristic, not a semantic circuit finder."""

import torch

POSITIVE = (
    "debugging", "debugging code", "software bug investigation",
    "trace a program", "software engineering",
)
NEUTRAL = ("rain", "coffee", "quiet Sunday", "gardening", "music")
SUPPORTED_CONCEPT = "debugging"


def embedding_direction(tokenizer, embedding_weight: torch.Tensor, concept: str) -> torch.Tensor:
    if concept != SUPPORTED_CONCEPT:
        raise ValueError("Only --concept debugging is supported by this experimental contrast set.")
    if embedding_weight.ndim != 2 or embedding_weight.shape[1] != 896:
        raise ValueError("Expected token embeddings with hidden dimension 896.")

    def phrase_mean(phrase: str) -> torch.Tensor:
        ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"Tokenizer produced no tokens for {phrase!r}.")
        if min(ids) < 0 or max(ids) >= embedding_weight.shape[0]:
            raise ValueError("Tokenizer IDs are outside the embedding table.")
        return embedding_weight[ids].float().mean(dim=0)

    with torch.no_grad():
        positive = torch.stack([phrase_mean(p) for p in POSITIVE]).mean(dim=0)
        neutral = torch.stack([phrase_mean(p) for p in NEUTRAL]).mean(dim=0)
        direction = positive - neutral
        norm = torch.linalg.vector_norm(direction)
        if not torch.isfinite(norm) or norm.item() <= 1e-12:
            raise ValueError("The concept contrast has a zero or nonfinite norm.")
        return direction / norm
