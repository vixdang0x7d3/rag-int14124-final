import re
from typing import Counter

import tiktoken
from fuzzywuzzy import fuzz, process

from chromadb.utils import embedding_functions

# from chromaDB evaluation repo


def find_query_despite_whitespace(document, query):
    # Normalize spaces and newlines in the query
    normalized_query = re.sub(r"\s+", " ", query).strip()

    # Create a regex pattern from the normalized query to match any whitespace characters between words
    pattern = r"\s*".join(re.escape(word) for word in normalized_query.split())

    # Compile the regex to ignore case and search for it in the document
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)

    if match:
        return document[match.start() :, match.end()], match.start(), match.end()
    else:
        return None


def rigorous_document_search(document: str, target: str):
    """
    This function performs a rigorous search of a target string within a document.
    It handles issues related to whitespaces, changes in grammar, and other minor text alterations.
    The function first checks for an exact match of the target in the document.
    If no exact match is found, it performs a raw search that account for variation in whitespace.
    If the raw search also fails, it splits the document into sentences and uses fuzzy matching
    to find the sentence that best matches the target.

    Args:
        document (str): The document in which to search for the target.
        target (str): The string to search for within the document.

    Returns:
        tuple: A tuple containing the best match found in the document, its start index, its end index.
        If no match is found, returns None
    """

    if target.endswith("."):
        target = target[:-1]

    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index
    else:
        raw_search = find_query_despite_whitespace(document, target)
        if raw_search is not None:
            return raw_search

    sentences = re.split(r"[.!?]\s*|\n", document)

    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)

    if not best_match or best_match[1] < 98:
        return None

    reference = best_match[0]

    start_index = document.find(reference)
    end_index = start_index + len(reference)

    return reference, start_index, end_index


def get_sentencetransformer_embedding_function(
    model_name: str = "all-MiniLM-L6-v2", device="cpu"
):
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name, device=device
    )
    return embedding_function


def openai_token_count(string: str) -> int:
    """Returns the number of tokens in a text string"""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def sum_of_ranges(ranges):
    return sum(end - start for start, end in ranges)


def union_ranges(ranges):
    # Sort ranges based on the starting index
    sorted_ranges = sorted(ranges, key=lambda x: x[0])

    # Initialize with the first range
    merged_ranges = [sorted_ranges[0]]

    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged_ranges[-1]

        # Check if the current range overlaps or is contiguous with the last range in the merged list
        if current_start <= last_end:
            # Merge the two ranges
            merged_ranges[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap, add the current range as new
            merged_ranges.append((current_start, current_end))

    return merged_ranges


def intersect_two_ranges(range1, range2):
    # Unpack the ranges
    start1, end1 = range1
    start2, end2 = range2

    # Get the maximum of the starting indices and the minimum of the ending indices
    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)

    if intersect_start <= intersect_end:
        return intersect_start, intersect_end
    else:
        return None


def difference(ranges, target):
    """
    Takes a set of ranges and a target range, and returns the difference.

    Args:
        ranges (list[tuple]): A list of tuples representing ranges. Each tuple is (a, b) where a <= b.
        target (tuple): A tuple representing a target range (c, d) where  c <= d.

    Returns:
        List of tuples representing ranges after removing the segments that overlap with the target range.
    """

    results = []

    target_start, target_end = target

    for start, end in ranges:
        if end < target_start or start > target_end:
            results.append((start, target_start))
        elif start < target_start and end > target_end:
            results += [(start, target_start), (target_end, end)]
        elif start < target_start:
            results.append((start, target_start))
        elif end > target_end:
            results.append((target_end, end))

    return results


def find_target_in_document(document, target):
    start_index = document.find(target)
    if start_index == -1:
        return None
    end_index = start_index + len(target)

    return start_index, end_index


# from OG rag full pipeline evaluation


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
