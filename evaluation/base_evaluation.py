import os
import json
import pandas as pd

from chromadb.utils import embedding_functions
import chromadb

from typing import Callable
from utils import rigorous_document_search


class GeneralDatasetCollection:
    def __init__(
        self, question_csv_path: str, chroma_db_path=None, corpora_id_paths=None
    ) -> None:
        self.corpora_id_paths = corpora_id_paths
        self.questions_csv_path = question_csv_path
        self.corpus_list = []

        self._load_questions_df()

        if chroma_db_path is not None:
            self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        else:
            self.chroma_client = chromadb.EphemeralClient()

        self.is_general = False

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
        metadatas = []

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
            metadatas.extend(current_metadatas)

        return documents, metadatas

    def _chunker_to_document_collection(
        self,
        chunker: Callable,
        embbeding_function: chromadb.EmbeddingFunction,
        chroma_db_path: str | None,
        collection_name: str | None = None,
    ):
        collection = None

        if chroma_db_path is not None:
            try:
                chunk_client = chromadb.PersistentClient(path=chroma_db_path)
                collection = chunk_client.create_collection(
                    collection_name,  # type: ignore
                    embedding_function=embbeding_function,
                    metadata={"hnsw:search_ef": 50},
                )

                print(f"Created collection: {collection_name}")
            except Exception as e:
                print(f"Failed to create collection: {e}")
                print("Fall back to default setting...")
                pass

        # Fallback to default embedding setting
        collection_name = "chunks_default"
        if collection is None:
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception:
                pass

            collection = self.chroma_client.create_collection(
                collection_name,
                embedding_function=embbeding_function,
                metadata={"hnsw:search_ef": 50},
            )

        docs, metas = self._get_chunks_and_metadata(chunker)

        BATCH_SIZE = 500
        for i in range(0, len(docs), BATCH_SIZE):
            batch_docs = docs[i : i + BATCH_SIZE]
            batch_metas = metas[i : i + BATCH_SIZE]
            batch_ids = [str(j) for j in range(i, i + len(batch_docs))]
            collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)

        return collection

    def _question_df_to_collection(
        self,
        embedding_function,
        chroma_db_path: str | None,
        collection_name: str = "default",
    ):
        self._load_questions_df()
        question_collection = None
