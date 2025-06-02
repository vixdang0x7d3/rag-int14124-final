import os
import json
from chromadb.utils import embedding_functions
import pandas as pd

import chromadb

from typing import Callable

from utils import rigorous_document_search


class BaseDatasetCollection:
    def __init__(
        self, question_csv_path: str, chroma_fallback_path=None, corpora_id_paths=None
    ) -> None:
        self.corpora_id_paths = corpora_id_paths
        self.questions_csv_path = question_csv_path
        self.corpus_list = []

        self._load_questions_df()

        if chroma_fallback_path is not None:
            # If a path is provided, save persists fall-back collections to a database in that path
            self.chroma_client = chromadb.PersistentClient(path=chroma_fallback_path)
        else:
            # Use a in-mem database
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
        """
        This function accepts a chunking algorithm, use it to split text
        then compute metadata for each chunk.
        """

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
        chroma_db_path: str | None = None,
        collection_name: str | None = None,
    ):
        collection = None

        if chroma_db_path is not None and collection_name is not None:
            try:
                chunk_client = chromadb.PersistentClient(path=chroma_db_path)
                collection = chunk_client.create_collection(
                    collection_name,
                    embedding_function=embbeding_function,
                    metadata={"hnsw:search_ef": 50},
                )

                print(f"Created collection: {collection_name}")
            except Exception as e:
                print(f"Failed to create collection: {e}")
                print("Using fallback configuration...")
                pass

        # if no chroma_db_path is not provided or creation fails
        # use fall-back configuration
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
        chroma_db_path: str | None = None,
        collection_name: str | None = None,
    ):
        self._load_questions_df()

        questions_collection = None

        if chroma_db_path is not None and collection_name is not None:
            try:
                questions_client = chromadb.PersistentClient(path=chroma_db_path)
                questions_collection = questions_client.create_collection(
                    collection_name,
                    embedding_function=embedding_function,
                    metadata={"hnsw:search_ef": 50},
                )
            except Exception as e:
                print(f"Failed to create collection: {e}")
                print("Use fallback configuration...")
                pass

        collection_name = "questions_default"

        if questions_collection is None:
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception:
                pass

            questions_collection = self.chroma_client.create_collection(
                collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:search_ef": 50},
            )

        questions_collection.add(
            documents=self.questions_df["question"].tolist(),
            metadatas=[
                {"corpus_id": x} for x in self.questions_df["corpus_id"].tolist()
            ],
            ids=[str(i) for i in self.questions_df.index],
        )

        return questions_collection

    def get_collections(
        self,
        chunker,
        embedding_function: chromadb.EmbeddingFunction | None,
        db_to_save_chunks: str | None = None,
        db_to_save_questions: str | None = None,
    ):
        """
        If caller provides database paths:
        Creates clients to connect to chunk database and questions database.
        Resolve collection name for each of them and try to fetch embeddings from databases.
        If fails, it will call *_to_collection's to create new collections and return them.
        Otherwise, the method will switch to a in-memory client.
        """

        if embedding_function is None:
            embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )

        collection = None
        if db_to_save_chunks is not None:
            chunk_size = (
                chunker.__chunk_size if hasattr(chunker, "__chunk_size") else "0"
            )

            chunk_overlap = (
                chunker.__chunk_overlap if hasattr(chunker, "chunk_overlap") else "0"
            )

            embedding_function_name = embedding_function.__class__.__name__

            if embedding_function_name == "SentenceTransformerEmbeddingFunction":
                embedding_function_name = "SentEmbFunc"

            model_name = (
                embedding_function.model_name  # type: ignore
                if hasattr(embedding_function, "model_name")
                else "unknown"
            )

            collection_name = (
                "chunks_"
                + embedding_function_name
                + "_"
                + model_name
                + "_"
                + chunker.__class__.__name__
                + "_"
                + str(int(chunk_size))
                + "_"
                + str(int(chunk_overlap))
            )

            try:
                chunk_client = chromadb.PersistentClient(path=collection_name)
                collection = chunk_client.get_collection(
                    collection_name, embedding_function=embedding_function
                )
            except Exception:
                collection = self._chunker_to_document_collection(
                    chunker,
                    embedding_function,
                    chroma_db_path=db_to_save_chunks,
                    collection_name=collection_name,
                )

        if collection is None:
            collection = self._chunker_to_document_collection(
                chunker, embedding_function
            )

        questions_collection = None

        if self.is_general:
            if db_to_save_questions is not None:
                embedding_function_name = embedding_function.__class__.__name__
                if embedding_function_name == "SentenceTransformerEmbeddingFunction":
                    embedding_function_name = "SentEmbFunc"

                model_name = (
                    embedding_function.model_name  # type: ignore
                    if hasattr(embedding_function, "model_name")
                    else "unknown"
                )

                collection_name = (
                    "questions_" + embedding_function_name + "_" + model_name
                )

                try:
                    question_client = chromadb.PersistentClient(
                        path=db_to_save_questions
                    )
                    questions_collection = question_client.get_collection(
                        collection_name, embedding_function=embedding_function
                    )
                except Exception:
                    questions_collection = self._question_df_to_collection(
                        embedding_function, db_to_save_questions, collection_name
                    )

        if not self.is_general or questions_collection is None:
            questions_collection = self._question_df_to_collection(embedding_function)

        return collection, questions_collection
