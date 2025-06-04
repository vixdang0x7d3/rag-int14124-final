import os
import json
import pandas as pd
import numpy as np


from utils import (
    difference,
    intersect_two_ranges,
    sum_of_ranges,
    union_ranges,
)

from chromadb import Collection


class Evaluation:
    def __init__(
        self,
        questions_csv_path: str,
        collection: Collection,
        questions_collection: Collection,
    ):
        self._collection = collection
        self._questions_collection = questions_collection

        self._questions_csv_path = questions_csv_path
        self._load_questions_df()

    def _load_questions_df(self):
        if os.path.exists(self._questions_csv_path):
            self._questions_df = pd.read_csv(self._questions_csv_path)
            self._questions_df["references"] = self._questions_df["references"].apply(
                json.loads
            )
        else:
            self._questions_df = pd.DataFrame(
                columns=["question", "references", "corpus_id"]  # type: ignore
            )

    def _full_precision_score(self, chunk_metadatas):
        ioc_scores = []
        recall_scores = []

        highlighted_chunks_count = []

        for _, row in self._questions_df.iterrows():
            _ = row["question"]
            references = row["references"]
            corpus_id = row["corpus_id"]

            ioc_score = 0
            numerator_sets = []
            denominator_chunks_sets = []
            unused_highlights = [(x["start_index"], x["end_index"]) for x in references]

            highlighted_chunk_count = 0

            for metadata in chunk_metadatas:
                chunk_start, chunk_end, chunk_corpus_id = (
                    metadata["start_index"],
                    metadata["end_index"],
                    metadata["corpus_id"],
                )

                if chunk_corpus_id != corpus_id:
                    continue

                contains_highlight = False

                for ref_obj in references:
                    _ = ref_obj["content"]
                    ref_start, ref_end = (
                        int(ref_obj["start_index"]),
                        int(ref_obj["end_index"]),
                    )
                    intersection = intersect_two_ranges(
                        (chunk_start, chunk_end), (ref_start, ref_end)
                    )

                    if intersection is not None:
                        contains_highlight = True

                        # Remove intersection from unused highlight
                        unused_highlights = difference(unused_highlights, intersection)

                        # Add intersection to numerator sets
                        numerator_sets = union_ranges([intersection] + numerator_sets)

                        # Add chunk to denominator sets
                        denominator_chunks_sets = union_ranges(
                            [(chunk_start, chunk_end)] + denominator_chunks_sets
                        )

                if contains_highlight:
                    highlighted_chunk_count += 1

            highlighted_chunks_count.append(highlighted_chunk_count)

            # Combine unused highlights and chunks for final denominator
            denominator_sets = union_ranges(denominator_chunks_sets + unused_highlights)

            # Calculate ioc_scores if there are numerator_sets
            if numerator_sets:
                ioc_score = sum_of_ranges(numerator_sets) / sum_of_ranges(
                    denominator_sets
                )

            ioc_scores.append(ioc_score)
            recall_score = 1 - (
                sum_of_ranges(unused_highlights)
                / sum_of_ranges(
                    [(x["start_index"], x["end_index"]) for x in references]
                )
            )
            recall_scores.append(recall_score)

        return ioc_scores, recall_scores

    def _scores_from_dataset_and_retrieval(
        self, question_metadatas, highlighted_chunks_count
    ):
        iou_scores = []
        recall_scores = []
        precision_scores = []

        for (_, row), highlighted_chunk_count, metadatas in zip(
            self._questions_df.iterrows(), highlighted_chunks_count, question_metadatas
        ):
            _ = row["question"]
            references = row["references"]
            corpus_id = row["corpus_id"]

            numerator_sets = []
            denominator_chunks_sets = []
            unused_highlights = [(x["start_index"], x["end_index"]) for x in references]

            for metadata in metadatas[:highlighted_chunk_count]:
                chunk_start, chunk_end, chunk_corpus_id = (
                    metadata["start_index"],
                    metadata["end_index"],
                    metadata["corpus_id"],
                )

                if chunk_corpus_id != corpus_id:
                    continue

                for ref_obj in references:
                    _ = ref_obj["content"]

                    ref_start, ref_end = (
                        int(ref_obj["start_index"]),
                        int(ref_obj["end_index"]),
                    )

                    intersection = intersect_two_ranges(
                        (chunk_start, chunk_end), (ref_start, ref_end)
                    )

                    if intersection is not None:
                        unused_highlights = difference(unused_highlights, intersection)

                        numerator_sets = union_ranges([intersection] + numerator_sets)

                        denominator_chunks_sets = union_ranges(
                            [(chunk_start, chunk_end)] + denominator_chunks_sets
                        )

            if numerator_sets:
                numerator_value = sum_of_ranges(numerator_sets)
            else:
                numerator_value = 0

            recall_denominator = sum_of_ranges(
                [(x["start_index"], x["end_index"]) for x in references]
            )

            precision_denominator = sum_of_ranges(
                [
                    (x["start_index"], x["end_index"])
                    for x in metadatas[:highlighted_chunk_count]
                ]
            )

            iou_denominator = precision_denominator + sum_of_ranges(unused_highlights)

            print(
                [
                    (x["start_index"], x["end_index"])
                    for x in metadatas[:highlighted_chunk_count]
                ]
            )

            # print(recall_denominator)
            # print(precision_denominator)
            # print(iou_denominator)
            # print("---")

            recall_score = numerator_value / recall_denominator
            recall_scores.append(recall_score)

            precision_score = (
                numerator_value / precision_denominator
                if precision_denominator > 0
                else 0.0
            )
            precision_scores.append(precision_score)

            iou_score = numerator_value / iou_denominator
            iou_scores.append(iou_score)

        return iou_scores, recall_scores, precision_scores

    def evaluate(self, retrieve: int = 5):
        questions_db = self._questions_collection.get(include=["embeddings"])

        questions_db_ids = [int(id) for id in questions_db["ids"]]
        embeddings = np.array(questions_db["embeddings"])

        id_embedding_pairs = [
            (id_val, emb) for id_val, emb in zip(questions_db_ids, embeddings.tolist())
        ]

        sorted_pairs = sorted(id_embedding_pairs, key=lambda x: x[0])
        _, sorted_embeddings = zip(*sorted_pairs)

        self._questions_df = self._questions_df.sort_index()

        brute_iou_scores, highlighted_chunks_count = self._full_precision_score(
            self._collection.get()["metadatas"]
        )

        if retrieve == -1:
            maximum_n = min(20, max(highlighted_chunks_count))
        else:
            highlighted_chunks_count = [retrieve] * len(highlighted_chunks_count)
            maximum_n = retrieve

        retrievals = self._collection.query(
            query_embeddings=list(sorted_embeddings), n_results=maximum_n
        )

        iou_scores, recall_scores, precision_scores = (
            self._scores_from_dataset_and_retrieval(
                retrievals["metadatas"], highlighted_chunks_count
            )
        )

        corpora_scores = {}

        for index, (_, row) in enumerate(self._questions_df.iterrows()):
            if row["corpus_id"] not in corpora_scores:
                corpora_scores[row["corpus_id"]] = {
                    "precision_omega_scores": [],
                    "iou_scores": [],
                    "recall_scores": [],
                    "precision_scores": [],
                }

            corpora_scores[row["corpus_id"]]["precision_omega_scores"].append(
                brute_iou_scores[index]
            )

            corpora_scores[row["corpus_id"]]["iou_scores"].append(iou_scores[index])
            corpora_scores[row["corpus_id"]]["recall_scores"].append(
                recall_scores[index]
            )
            corpora_scores[row["corpus_id"]]["precision_scores"].append(
                precision_scores[index]
            )

        brute_iou_mean = np.mean(brute_iou_scores)
        brute_iou_std = np.std(brute_iou_scores)

        recall_mean = np.mean(recall_scores)
        recall_std = np.std(recall_scores)

        iou_mean = np.mean(iou_scores)
        iou_std = np.std(iou_scores)

        precision_mean = np.mean(precision_scores)
        precision_std = np.std(precision_scores)

        return {
            "corpora_scores": corpora_scores,
            "iou_mean": iou_mean,
            "iou_std": iou_std,
            "recall_mean": recall_mean,
            "recall_std": recall_std,
            "precision_omega_mean": brute_iou_mean,
            "precision_omega_std": brute_iou_std,
            "precision_mean": precision_mean,
            "precision_std": precision_std,
        }
