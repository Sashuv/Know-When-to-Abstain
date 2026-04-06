from helpers import filter_split, clear_embed_cache
from config import SamplerConfig
from sampler import adaptive_sample, build_prediction_set
from calibration import check_coverage, run_calibration


DEFAULT_CONFIG = SamplerConfig()

def evaluate(
    DATASETS,
    dataset_name: str,
    alpha:        float,
    q_hat:        float,
    config:       SamplerConfig = DEFAULT_CONFIG,
    max_samples:  int           = 500,
) -> dict:
    """
    Run AdaptiveCP on the TEST split and compute all metrics.

    Metrics (matching LofreeCP Table 1/2):
      ECR  = Empirical Coverage Rate (%) — fraction of questions where
             gold answer is in prediction set OR system abstained
             Target: >= (1 - alpha) * 100

      SSC  = Selective Set Coverage (%) — coverage among NON-abstained
             questions only. Measures quality of answers given.
             Higher = better. Abstentions excluded from denominator.

      APSS = Average Prediction Set Size — mean size of non-empty
             prediction sets. 0 for abstentions.
             Lower = more efficient. Computed over ALL questions.

      API_CALLS = Average number of model calls per question.
             AdaptiveCP's key efficiency claim vs LofreeCP's fixed 20-30.

    Args:
        q_hat: calibrated threshold from run_calibration()
               Must be calibrated on calibration split, NOT test split.
    """
    test_samples = filter_split(DATASETS[dataset_name], "test")

    if len(test_samples) > max_samples:
        import random
        test_samples = random.sample(test_samples, max_samples)

    print(f"\nEvaluating {dataset_name} | alpha={alpha} | q_hat={q_hat:.4f} "
          f"| n={len(test_samples)}")

    n_covered       = 0   # for ECR
    n_ssc_covered   = 0   # for SSC (numerator)
    n_ssc_total     = 0   # for SSC (denominator — non-abstained only)
    total_set_size  = 0   # for APSS (sum over ALL questions)
    total_api_calls = 0   # for avg_api_calls

    results = []

    for i, sample in enumerate(test_samples):
        clear_embed_cache()

        result = adaptive_sample(sample.question, config=config)
        pred_set = build_prediction_set(result, q_hat)

        covered = check_coverage(pred_set, sample.gold_answers)
        abstained = result["abstain"] or (result["n_score"] is not None
                                          and result["n_score"] > q_hat)

        # ECR: covered = True for both correct answers AND abstentions
        if covered:
            n_covered += 1

        # SSC: only count non-abstained questions
        if not abstained:
            n_ssc_total += 1
            if covered:
                n_ssc_covered += 1

        # APSS: set size for ALL questions (abstentions contribute 0)
        total_set_size += len(pred_set)

        # API calls = n_samples used by adaptive sampler
        total_api_calls += result["n_samples"]

        result["question"]     = sample.question
        result["gold_answers"] = sample.gold_answers
        result["pred_set"]     = pred_set
        result["covered"]      = covered
        result["abstained"]    = abstained
        results.append(result)

        if (i + 1) % 100 == 0:
            ecr_so_far = n_covered / (i + 1) * 100
            print(f"  [{i+1:4d}/{len(test_samples)}] ECR={ecr_so_far:.1f}%")

    n = len(test_samples)
    ecr       = n_covered / n * 100
    ssc       = (n_ssc_covered / n_ssc_total * 100) if n_ssc_total > 0 else 0.0
    apss      = total_set_size / n
    api_calls = total_api_calls / n
    abstain_rate = sum(1 for r in results if r["abstained"]) / n * 100

    return {
        "dataset":      dataset_name,
        "alpha":        alpha,
        "q_hat":        q_hat,
        "n":            n,
        "ECR":          round(ecr, 1),
        "SSC":          round(ssc, 1),
        "APSS":         round(apss, 2),
        "api_calls":    round(api_calls, 1),
        "abstain_rate": round(abstain_rate, 1),
        "results":      results,
    }


def run_full_evaluation(
    DATASETS,
    dataset_name:   str,
    alphas:         list,
    lofree_ref:     dict,
    cal_samples:    int = 500,
    test_samples:   int = 500,
) -> list:
    """
    For each alpha:
      1. Calibrate on calibration split → q_hat
      2. Evaluate on test split → ECR, SSC, APSS, api_calls
      3. Print comparison row vs LofreeCP

    Returns list of result dicts (one per alpha).
    """
    all_results = []

    print(f"\n{'='*70}")
    print(f"FULL EVALUATION: {dataset_name.upper()}")
    print(f"{'='*70}")

    for alpha in sorted(alphas):
        print(f"\n--- alpha={alpha} (target ECR >= {(1-alpha)*100:.0f}%) ---")

        # Step 1: Calibrate
        cal = run_calibration(
            DATASETS,
            dataset_name,
            alpha       = alpha,
            config      = DEFAULT_CONFIG,
            max_samples = cal_samples,
        )
        q_hat = cal["q_hat"]

        # Step 2: Evaluate on test split
        eval_result = evaluate(
            DATASETS,
            dataset_name,
            alpha       = alpha,
            q_hat       = q_hat,
            config      = DEFAULT_CONFIG,
            max_samples = test_samples,
        )
        eval_result["cal_coverage"] = cal["coverage"]
        all_results.append(eval_result)

        # Step 3: Print comparison
        ref = lofree_ref.get(alpha, {})
        ecr_ok = "✓" if eval_result["ECR"] >= (1 - alpha) * 100 else "✗"

        print(f"\n  Results vs LofreeCP (alpha={alpha}):")
        print(f"  {'Metric':<12} {'AdaptiveCP':>12} {'LofreeCP':>12} {'Delta':>10}")
        print(f"  {'-'*48}")
        print(f"  {'ECR '+ecr_ok:<12} {eval_result['ECR']:>11.1f}% "
              f"{ref.get('ECR', '-'):>11}  "
              f"{eval_result['ECR'] - ref.get('ECR', eval_result['ECR']):>+9.1f}")
        print(f"  {'SSC ↑':<12} {eval_result['SSC']:>11.1f}% "
              f"{ref.get('SSC', '-'):>11}  "
              f"{eval_result['SSC'] - ref.get('SSC', eval_result['SSC']):>+9.1f}")
        print(f"  {'APSS ↓':<12} {eval_result['APSS']:>12.2f} "
              f"{ref.get('APSS', '-'):>11}  "
              f"{eval_result['APSS'] - ref.get('APSS', eval_result['APSS']):>+9.2f}")
        print(f"  {'API calls':<12} {eval_result['api_calls']:>12.1f} "
              f"{'~20-30':>11}  {'(adaptive)':>10}")
        print(f"  {'Abstain%':<12} {eval_result['abstain_rate']:>11.1f}%")

    return all_results

def print_comparison_table(all_results: list, lofree_ref: dict,
                            dataset_name: str) -> None:
    """
    Print a clean comparison table matching LofreeCP's paper format.
    Also prints a LaTeX version for copy-paste into the paper.
    """
    print(f"\n{'='*75}")
    print(f"COMPARISON TABLE: {dataset_name.upper()}")
    print(f"Logit-Access: ✗ (black-box)   Model: Mistral-7B-Instruct-v0.3")
    print(f"{'='*75}")

    alphas = sorted(set(r["alpha"] for r in all_results))
    header = f"{'Method':<20}"
    for a in alphas:
        header += f"  α={a}"
    print(f"\n{header}")

    # ECR row
    row_ecr = f"{'ECR (target→)':<20}"
    for a in alphas:
        target = (1 - a) * 100
        r = next(r for r in all_results if r["alpha"] == a)
        ok = "✓" if r["ECR"] >= target else "✗"
        row_ecr += f"  {r['ECR']:.1f}{ok}"
    print(row_ecr)

    # Reference ECR
    ref_ecr = f"{'  LofreeCP ECR':<20}"
    for a in alphas:
        ref = lofree_ref.get(a, {})
        ref_ecr += f"  {ref.get('ECR', '-')}"
    print(ref_ecr)

    print()

    # SSC row
    row_ssc = f"{'SSC ↑':<20}"
    for a in alphas:
        r = next(r for r in all_results if r["alpha"] == a)
        ref = lofree_ref.get(a, {})
        delta = r["SSC"] - ref.get("SSC", r["SSC"])
        marker = "↑" if delta > 1 else ("↓" if delta < -1 else "~")
        row_ssc += f"  {r['SSC']:.1f}{marker}"
    print(row_ssc)

    ref_ssc = f"{'  LofreeCP SSC':<20}"
    for a in alphas:
        ref = lofree_ref.get(a, {})
        ref_ssc += f"  {ref.get('SSC', '-')}"
    print(ref_ssc)

    print()

    # APSS row
    row_apss = f"{'APSS ↓':<20}"
    for a in alphas:
        r = next(r for r in all_results if r["alpha"] == a)
        ref = lofree_ref.get(a, {})
        delta = r["APSS"] - ref.get("APSS", r["APSS"])
        marker = "↓" if delta < -0.05 else ("↑" if delta > 0.05 else "~")
        row_apss += f"  {r['APSS']:.2f}{marker}"
    print(row_apss)

    ref_apss = f"{'  LofreeCP APSS':<20}"
    for a in alphas:
        ref = lofree_ref.get(a, {})
        ref_apss += f"  {ref.get('APSS', '-')}"
    print(ref_apss)

    print()

    # API calls row (new — LofreeCP doesn't report this)
    row_api = f"{'API calls ↓ (new)':<20}"
    for a in alphas:
        r = next(r for r in all_results if r["alpha"] == a)
        row_api += f"  {r['api_calls']:.1f}"
    print(row_api)
    print(f"{'  LofreeCP API calls':<20}  {'~25 (fixed)':>}")

    print(f"\n  ↑/↓/~ = better/worse/same vs LofreeCP")