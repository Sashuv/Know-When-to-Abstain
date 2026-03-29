from collections import Counter
import numpy as np


def _assign_split(i: int, n_total: int) -> str:
    """
    50% calibration / 25% validation / 25% test
    Deterministic given index and total.
    """
    frac = i / n_total
    if frac < 0.50:
        return "calibration"
    elif frac < 0.75:
        return "validation"
    else:
        return "test"

def _print_split_counts(samples: list) -> None:
    counts = Counter(s.split for s in samples)
    for split in ["calibration", "validation", "test"]:
        print(f"  {split}: {counts.get(split, 0)}")

def filter_split(samples: list, split: str) -> list:
    return [s for s in samples if s.split == split]


# Clean Response
def clean_response(response: str, max_tokens: int = 8) -> str:
    """
    Normalize a raw model response to a short clean phrase.

    Steps:
      1. Strip whitespace
      2. Truncate at sentence-ending punctuation or explanatory conjunctions
      3. Hard cap at max_tokens words
      4. Lowercase
    """
    response = response.strip()

    stop_chars = [".", "\n", ";"]
    stop_words = [" because", " since", " which", " who", " as ", " but "]

    for char in stop_chars:
        if char in response:
            response = response[:response.index(char)].strip()

    for word in stop_words:
        if word in response.lower():
            idx = response.lower().index(word)
            response = response[:idx].strip()

    tokens = response.split()
    if len(tokens) > max_tokens:
        response = " ".join(tokens[:max_tokens])

    return response.lower().strip()



_embed_cache: dict = {}
def embed_cached(text: str) -> np.ndarray:
    """Embed text using MiniLM with caching. Returns L2-normalized vector."""
    key = text.strip().lower()
    if key not in _embed_cache:
        emb = ENCODER.encode(
            key,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        _embed_cache[key] = emb
    return _embed_cache[key]

def clear_embed_cache():
    """Call between questions to prevent unbounded memory growth."""
    _embed_cache.clear()