import math
import json
import time
from sampler import adaptive_sample, check_coverage
from config import SamplerConfig
from helpers import filter_split, embed_cached, clear_embed_cache
DEFAULT_CONFIG = SamplerConfig()

def compute_qhat(n_scores: list, alpha: float) -> float:
    """
    Compute the conformal prediction threshold q_hat.

    Standard split-CP formula:
        position = ceil((n + 1) * (1 - alpha))
        q_hat = n_scores_sorted[position - 1]   (1-indexed)

    If position > n (can happen with small n or large alpha),
    q_hat = infinity — meaning every question gets a prediction set.
    This is the correct conservative fallback.

    Args:
        n_scores: list of nonconformity scores from calibration
                  (abstentions excluded)
        alpha:    error rate (0.1 = 90% coverage target)

    Returns:
        float q_hat threshold
    """
    n = len(n_scores)

    if n == 0:
        print("  WARNING: no scored calibration examples. q_hat = inf.")
        return float("inf")

    sorted_scores = sorted(n_scores)
    position      = math.ceil((n + 1) * (1 - alpha))

    if position > n:
        print(f"  NOTE: position={position} > n={n}. q_hat = inf (conservative).")
        return float("inf")

    q_hat = sorted_scores[position - 1]
    return q_hat


def run_calibration(
    model,
    tokenizer,
    DATASETS,
    dataset_name: str,
    alpha:        float         = 0.1,
    config:       SamplerConfig = DEFAULT_CONFIG,
    max_samples:  int           = 500,
    save_path:    str           = None,
) -> dict:
    """
    Run conformal calibration on the calibration split of a dataset.

    Args:
        dataset_name: "triviaqa" | "webq" | "mmlu"
        alpha:        error rate. 0.1 = target 90% coverage.
        config:       SamplerConfig for adaptive_sample
        max_samples:  cap on calibration set size (500 is sufficient)
        save_path:    if given, save results as JSON to this path

    Returns calibration dict with:
        q_hat       — the conformal threshold
        n_scores    — all nonconformity scores collected
        abstain_rate — fraction that abstained during calibration
        results     — full per-question result list (for diagnostics)

    Conformal quantile formula (standard split-CP):
        q_hat = quantile of n_scores at level ceil((n+1)(1-alpha)) / n
        This guarantees marginal coverage >= 1-alpha on test data.

    Abstentions are excluded from n_scores — they don't contribute
    to the threshold because they'll never be compared against q_hat
    at test time (abstentions bypass the threshold entirely).
    """
    samples = filter_split(DATASETS[dataset_name], "calibration")

    # Cap at max_samples for speed
    if len(samples) > max_samples:
        import random
        samples = random.sample(samples, max_samples)

    print(f"\nCalibrating on {dataset_name} "
          f"(n={len(samples)}, alpha={alpha})")
    print(f"Config: {config}")
    print("-" * 55)

    results      = []
    n_scores     = []
    n_abstained  = 0
    t_start      = time.time()

    for i, sample in enumerate(samples):
        clear_embed_cache()

        result = adaptive_sample(model=model, tokenizer=tokenizer, question=sample.question, config=config)

        result["question"]     = sample.question
        result["gold_answers"] = sample.gold_answers
        result["source"]       = sample.source

        # Coverage check (for diagnostics — not used to compute q_hat)
        covered = check_coverage(
            result["cleaned"] if not result["abstain"] else [],
            sample.gold_answers,
        )
        result["covered"] = covered

        if result["abstain"]:
            n_abstained += 1
        else:
            n_scores.append(result["n_score"])

        results.append(result)

        # Progress log every 50 questions
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            elapsed  = time.time() - t_start
            per_q    = elapsed / (i + 1)
            eta      = per_q * (len(samples) - i - 1)
            coverage = sum(r["covered"] for r in results) / len(results)
            print(f"  [{i+1:4d}/{len(samples)}] "
                  f"abstain_rate={n_abstained/(i+1)*100:.1f}%  "
                  f"coverage={coverage*100:.1f}%  "
                  f"eta={eta/60:.1f}min")

    # Compute q_hat using standard split-CP formula
    q_hat = compute_qhat(n_scores, alpha)

    abstain_rate = n_abstained / len(samples)
    coverage     = sum(r["covered"] for r in results) / len(results)

    cal_dict = {
        "dataset":      dataset_name,
        "alpha":        alpha,
        "n_total":      len(samples),
        "n_scored":     len(n_scores),
        "n_abstained":  n_abstained,
        "abstain_rate": abstain_rate,
        "q_hat":        q_hat,
        "n_scores":     n_scores,
        "coverage":     coverage,
        "config":       {
            "batch_size":  config.batch_size,
            "max_batches": config.max_batches,
            "eps":         config.eps,
            "temperature": config.temperature,
            "min_batches": config.min_batches,
        },
        "results":      results,
    }

    print(f"\nCalibration complete — {dataset_name}")
    print(f"  n_scored     : {len(n_scores)}")
    print(f"  n_abstained  : {n_abstained} ({abstain_rate*100:.1f}%)")
    print(f"  q_hat        : {q_hat:.4f}")
    print(f"  cal coverage : {coverage*100:.1f}%  (diagnostic only)")

    if save_path:
        # Save everything except full result objects (too large for JSON)
        save_dict = {k: v for k, v in cal_dict.items() if k != "results"}
        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=2)
        print(f"  Saved to {save_path}")

    return cal_dict