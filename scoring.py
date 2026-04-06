import numpy as np
from model import load_sentence_encoder
from collections import Counter
from config import UNCERTAINTY_TOKENS

ENCODER = None

def get_encoder():
    global ENCODER
    if ENCODER is None:
        ENCODER = load_sentence_encoder()
    return ENCODER


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

# Compute Stability
def compute_stability(responses: list) -> float:
    """
    Mean pairwise cosine similarity across all response pairs,
    computed in MiniLM sentence embedding space.

    Captures semantic equivalence regardless of surface phrasing:
      "shakespeare" ≈ "william shakespeare"         → high similarity
      "decline & migration" ≈ "fall of roman empire" → high similarity
      "shakespeare" vs "napoleon"                    → low similarity

    Returns float in [0, 1].
    """
    cleaned = [clean_response(r) for r in responses]

    if len(cleaned) == 1:
        return 1.0

    # Batch-embed all unique texts in one forward pass
    unique_texts = list(set(cleaned))
    ENCODER = get_encoder()
    embeddings_matrix = ENCODER.encode(
        unique_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    embed_map = {text: embeddings_matrix[i] for i, text in enumerate(unique_texts)}

    scores = []
    for i in range(len(cleaned)):
        for j in range(i + 1, len(cleaned)):
            sim = float(np.dot(embed_map[cleaned[i]], embed_map[cleaned[j]]))
            scores.append(max(0.0, sim))

    return float(np.mean(scores))

# PLateau Stop
def plateau_stop(history: list, eps: float = 0.03,
                 min_stability: float = 0.70) -> bool:
    """
    Return True if stability has converged AND is high enough to act on.

    Two conditions must both hold:
      1. Convergence: last two consecutive deltas both < eps
      2. Quality floor: current stability >= min_stability

    Without condition 2, a low-stability trajectory that flattens
    (e.g. 0.45 → 0.43 → 0.42, all deltas < eps) would incorrectly
    trigger a plateau and return a useless prediction set.

    min_stability=0.70 means the model must agree with itself on
    at least ~70% of response pairs to commit to a prediction set.
    """
    if len(history) < 3:
        return False

    delta_1 = abs(history[-1] - history[-2])
    delta_2 = abs(history[-2] - history[-3])

    converged      = delta_1 < eps and delta_2 < eps
    stable_enough  = history[-1] >= min_stability

    return converged and stable_enough

# Check if the response is uncertain
def is_uncertainty_response(response: str) -> bool:
    """Return True if response expresses ignorance rather than a fact."""
    r = response.lower().strip()
    return any(token in r for token in UNCERTAINTY_TOKENS)

# WOOPP
def responses_are_substantive(cleaned_responses: list,
                               uncertainty_threshold: float = 0.6) -> bool:
    """
    Return True if the majority of responses are substantive answers.

    If more than uncertainty_threshold fraction are uncertainty expressions,
    treat the response set as non-substantive → force abstention.
    """
    if not cleaned_responses:
        return False

    n_uncertain = sum(1 for r in cleaned_responses
                      if is_uncertainty_response(r))
    return (n_uncertain / len(cleaned_responses)) < uncertainty_threshold


def token_f1(a: str, b: str) -> float:
    """
    Token-level F1 with multiset overlap.
    Used in check_coverage for matching against gold answers.
    NOT used in compute_stability.
    """
    a_tokens = Counter(a.lower().split())
    b_tokens = Counter(b.lower().split())

    if not a_tokens or not b_tokens:
        return 0.0

    overlap   = sum((a_tokens & b_tokens).values())
    precision = overlap / sum(a_tokens.values())
    recall    = overlap / sum(b_tokens.values())

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)