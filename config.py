from dataclasses import dataclass, field


@dataclass
class QASample:
    question_id: str
    question:    str
    gold_answers: list   # list[str], always
    source:      str     # "triviaqa" | "webq" | "mmlu"
    split:       str     # "calibration" | "validation" | "test"

@dataclass
class SamplerConfig:
    batch_size:  int   = 3     # responses per batch
    max_batches: int   = 6     # hard budget = 18 samples max
    eps:         float = 0.03  # convergence tolerance
    temperature: float = 0.9   # sampling temperature
    min_batches: int   = 2     # never stop before 2 batches

# Uncertainity Tokens
UNCERTAINTY_TOKENS = {
    "unknown", "uncertain", "unclear", "unspecified", "unavailable",
    "undetermined", "unverified", "undefined", "unanswerable",
    "insufficient", "incomplete", "inconclusive",
    "no data", "no record", "no records", "no reliable", "no precise",
    "no accurate", "no specific", "no consensus", "no information",
    "not known", "not available", "not recorded", "not specified",
    "not enough", "not clear", "not certain", "not determined",
    "varies", "vary", "variable", "disputed", "debated", "contested",
    "estimates", "estimate", "approximat", "roughly", "around",
    "cannot", "can't", "could not", "impossible to",
    "sources differ", "sources vary", "sources disagree",
    "limited records", "limited data", "limited historical",
    "lack of", "lacking", "absence of",
    "historians disagree", "scholars disagree",
}

COVERAGE_SIM_THRESHOLD = 0.65

MMLU_SUBJECTS = [
    "high_school_mathematics",
    "college_medicine",
    "high_school_world_history",
    "professional_law",
    "abstract_algebra",
]