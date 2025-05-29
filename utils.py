import re
from typing import Counter


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation and extra whitespaces"""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def fix_white_space(text):
        return " ".join(text.split())

    def remove_punc(text):
        return text.lower()

    return fix_white_space(remove_articles(remove_punc(text.lower())))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def calcualate_exact_match(output_lns: list[str], reference_lns: list[str]):
    assert len(output_lns) == len(reference_lns)

    em = 0
    for hypo, pred in zip(output_lns, reference_lns):
        em += exact_match_score(hypo, pred)

    if len(output_lns) > 0:
        em /= len(output_lns)

    return em
