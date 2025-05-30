import os
import json
import pandas as pd

from utils import rigorous_document_search


class GeneralDatasetCollection:
    def __init__(
        self, question_csv_path: str, chroma_db_path=None, corpora_id_paths=None
    ) -> None:
        self.corpora_id_paths = corpora_id_paths
        self.questions_csv_path = question_csv_path
        self.corpus_list = []

        self._load_questions_df()

    def _load_questions_df(self):
        if os.path.exists(self.questions_csv_path):
            self.questions_df = pd.read_csv(self.questions_csv_path)
            self.questions_df["references"] = self.questions_df["references"].apply(
                json.loads
            )
        else:
            self.questions_df = pd.DataFrame(
                columns=["questions", "reference", "corpus_id"]  # type: ignore
            )

        self.corpus_list = self.questions_df

    def _get_chunks_and_metadata(self, splitter):
        documents = []
        metadata = []

        for corpus_id in self.corpus_list:
            corpus_path = corpus_id
            if self.corpora_id_paths is not None:
                corpus_path = self.corpora_id_paths[corpus_id]

            import platform

            if platform.system() == "Windows":
                with open(corpus_path, "r", encoding="utf-8") as f:
                    corpus = f.read()
            else:
                with open(corpus_path, "r") as f:
                    corpus = f.read()

            current_documents = splitter.split_text(corpus)
            current_metadatas = []

            for document in current_documents:
                start_index = end_index = -1
                try:
                    _, start_index, end_index = rigorous_document_search(  # type: ignore
                        corpus, document
                    )
                except Exception as e:
                    print(f"Error in finding {document}: {e}")

                current_metadatas.append(
                    {
                        "start_index": start_index,
                        "end_index": end_index,
                        "corpus_id": corpus_id,
                    }
                )

            documents.extend(current_documents)
            metadata.extend(current_metadatas)

        return documents, metadata

    def _chunker_to_collection(self): ...


if __name__ == "__main__":
    ...
