from datasets import load_dataset
from helpers import _assign_split, _print_split_counts, filter_split
from config import MMLU_SUBJECTS, QASample, SamplerConfig
import random

# Load TriviaQA dataset
def load_triviaqa(seed, n_total: int = 3000) -> list:
    """
    Load TriviaQA (rc.nocontext split). Normalize to QASample.
    n_total: how many examples to use across all splits.
    Returns list[QASample].
    """
    raw = load_dataset("trivia_qa", "rc.nocontext", split="train")
    raw = raw.shuffle(seed=seed).select(range(min(n_total, len(raw))))

    samples = []
    for i, row in enumerate(raw):
        # gold_answers: TriviaQA stores aliases under answer.aliases
        aliases = row["answer"]["aliases"]
        if not aliases:
            aliases = [row["answer"]["value"]]
        samples.append(QASample(
            question_id  = f"triviaqa_{i}",
            question     = row["question"],
            gold_answers = [a.strip().lower() for a in aliases],
            source       = "triviaqa",
            split        = _assign_split(i, n_total),
        ))

    print(f"TriviaQA loaded: {len(samples)} samples")
    _print_split_counts(samples)
    return samples

# Load Web Questions dataset
def load_webquestions(seed) -> list:
    """
    Load WebQuestions (train split only — test has no gold answers).
    Returns list[QASample].
    """
    raw = load_dataset("web_questions", split="train")
    raw = raw.shuffle(seed=seed)
    n_total = len(raw)

    samples = []
    for i, row in enumerate(raw):
        answers = [a.strip().lower() for a in row["answers"]]
        samples.append(QASample(
            question_id  = f"webq_{i}",
            question     = row["question"],
            gold_answers = answers,
            source       = "webq",
            split        = _assign_split(i, n_total),
        ))

    print(f"WebQuestions loaded: {len(samples)} samples")
    _print_split_counts(samples)
    return samples



# Load MMLU Dataset
def load_mmlu(seed, subjects: list = MMLU_SUBJECTS) -> list:
    """
    Load a selection of MMLU subjects. Converts MCQ to open-ended format.
    gold_answers = [correct_choice_text] (not the letter).
    Returns list[QASample].
    """
    LETTERS = ["A", "B", "C", "D"]
    all_samples = []

    for subject in subjects:
        raw = load_dataset("cais/mmlu", subject, split="test")
        for i, row in enumerate(raw):
            choices   = row["choices"]          # list of 4 strings
            answer_idx = row["answer"]           # int 0-3
            correct   = choices[answer_idx].strip().lower()

            # Build question string that includes the choices
            choices_text = "\n".join(
                f"{LETTERS[j]}) {choices[j]}" for j in range(len(choices))
            )
            question = f"{row['question']}\n{choices_text}"

            all_samples.append(QASample(
                question_id  = f"mmlu_{subject}_{i}",
                question     = question,
                gold_answers = [correct],
                source       = "mmlu",
                split        = "",   # assigned after pooling
            ))

    # Shuffle and assign splits after pooling all subjects
    random.shuffle(all_samples)
    n_total = len(all_samples)
    for i, s in enumerate(all_samples):
        s.split = _assign_split(i, n_total)

    print(f"MMLU loaded: {n_total} samples across {len(subjects)} subjects")
    _print_split_counts(all_samples)
    return all_samples