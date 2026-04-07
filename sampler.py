import numpy as np
from config import SamplerConfig, COVERAGE_SIM_THRESHOLD
from model import generate_batch
from scoring import compute_stability, plateau_stop, responses_are_substantive, token_f1
from helpers import filter_split, _print_split_counts, clean_response, embed_cached, clear_embed_cache
DEFAULT_CONFIG = SamplerConfig()

def adaptive_sample(
    model,
    tokenizer,
    question: str,
    config:   SamplerConfig = DEFAULT_CONFIG,
) -> dict:
    """
    Adaptively sample responses until plateau or budget exhausted.

    Returns a result dict. Key fields:

      abstain   (bool)  — True if no prediction set should be returned
      reason    (str)   — "plateau" | "uncertainty_responses" | "budget_exhausted"
      n_score   (float) —ta 1 - stability, or None if absining
      cleaned   (list)  — cleaned response strings
      history   (list)  — stability per batch
      n_batches (int)   — batches used
      n_samples (int)   — total samples used

    Three abstention conditions:
      1. Budget exhausted — plateau never reached
      2. Plateau reached but responses are all uncertainty expressions
         ("unknown", "uncertain", etc.) — model knows it doesn't know
      3. n_score > q_hat at test time (handled in build_prediction_set)
    """
    all_responses = []
    history       = []

    for batch_num in range(1, config.max_batches + 1):

        new_responses = generate_batch(
        model,
        tokenizer,
        question,
        n=config.batch_size,
        temperature=config.temperature,
        )
        all_responses.extend(new_responses)

        stability = compute_stability(all_responses)
        history.append(stability)

        if batch_num >= config.min_batches:
            if plateau_stop(history, eps=config.eps, min_stability= config.min_stability):

                cleaned = [clean_response(r) for r in all_responses]

                # Uncertainty gate
                if not responses_are_substantive(cleaned):
                    return {
                        "abstain":   True,
                        "reason":    "uncertainty_responses",
                        "responses": all_responses,
                        "cleaned":   cleaned,
                        "stability": stability,
                        "n_score":   None,
                        "n_batches": batch_num,
                        "n_samples": len(all_responses),
                        "history":   history,
                    }

                return {
                    "abstain":   False,
                    "reason":    "plateau",
                    "responses": all_responses,
                    "cleaned":   cleaned,
                    "stability": stability,
                    "n_score":   round(1.0 - stability, 6),
                    "n_batches": batch_num,
                    "n_samples": len(all_responses),
                    "history":   history,
                }

    # Budget exhausted
    cleaned = [clean_response(r) for r in all_responses]
    return {
        "abstain":   True,
        "reason":    "budget_exhausted",
        "responses": all_responses,
        "cleaned":   cleaned,
        "stability": history[-1] if history else 0.0,
        "n_score":   None,
        "n_batches": config.max_batches,
        "n_samples": len(all_responses),
        "history":   history,
    }

def build_prediction_set(result: dict, q_hat: float) -> list:
    """
    Return prediction set given sampler result and calibrated threshold q_hat.

    Returns [] (empty list) in three cases:
      - result["abstain"] is True (sampler abstained)
      - n_score > q_hat (question too uncertain per conformal threshold)

    Otherwise returns all unique cleaned responses.
    The conformal threshold q_hat controls set size — not frequency cutoffs.
    """
    if result["abstain"] or result["n_score"] is None:
        return []

    if result["n_score"] > q_hat:
        return []

    seen = []
    for r in result["cleaned"]:
        if r and r not in seen:
            seen.append(r)
    return seen


def check_coverage(prediction_set: list, gold_answers: list) -> bool:
    """
    Return True if any prediction is semantically similar to any gold answer.

    Uses MiniLM cosine similarity >= 0.65 as primary check.
    Falls back to token-F1 >= 0.5 for safety.
    Empty prediction set (abstention) always returns True.

    Why 0.65?
      - "eastern" vs "north american eastern time zone": cosine ~0.72 → covered ✓
      - "durbin" vs "dick durbin": cosine ~0.81 → covered ✓
      - "shakespeare" vs "napoleon": cosine ~0.35 → not covered ✓
      - "kansas" vs "united states of america": cosine ~0.52 → not covered ✓
        (this last one is genuinely wrong — Kansas ≠ USA)
    """
    if not prediction_set:
        return True

    for pred in prediction_set:
        if not pred:
            continue
        for gold in gold_answers:
            gold_clean = clean_response(gold, max_tokens=12)
            if not gold_clean:
                continue

            # Primary: semantic similarity
            try:
                sim = float(np.dot(embed_cached(pred), embed_cached(gold_clean)))
                if sim >= COVERAGE_SIM_THRESHOLD:
                    return True
            except Exception:
                pass

            # Fallback: token-F1
            if token_f1(pred, gold_clean) >= 0.5:
                return True

    return False