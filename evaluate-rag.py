import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Imports""")
    return


@app.cell
def _():
    import torch
    import re
    import pandas as pd

    from evaluation import GeneralEvalSet, Evaluation
    from chunker import BaseChunker, RecursiveTokenChunker

    from chromadb.utils import embedding_functions

    from pathlib import Path

    return (
        BaseChunker,
        Evaluation,
        GeneralEvalSet,
        Path,
        RecursiveTokenChunker,
        embedding_functions,
        re,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Setup code""")
    return


@app.cell
def _(torch):
    device = "cuda" if torch.cuda.is_available else "cpu"
    print(f"Using CUDA ? {'YES' if device == 'cuda' else 'NO'}")
    if device:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return (device,)


@app.function
def print_metrics(results):
    metrics = {
        "Recall": (results["recall_mean"], results["recall_std"]),
        "Precision": (results["precision_mean"], results["precision_std"]),
        "Precision Ω": (
            results["precision_omega_mean"],
            results["precision_omega_std"],
        ),
        "IoU": (results["iou_mean"], results["iou_std"]),
    }

    for metric, (mean, std) in metrics.items():
        print(f"{metric}: {mean:.4f} ± {std:.4f}")


@app.cell
def _(device, embedding_functions):
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2",
        device=device,
    )
    return (embedding_function,)


@app.cell
def _(GeneralEvalSet, Path):
    general_set = GeneralEvalSet(
        general_benchmark_path=Path("datasets/general_evaluation")
    )
    return (general_set,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Custom SentenceChunker""")
    return


@app.cell
def _(BaseChunker, re):
    class SentenceChunker(BaseChunker):
        def __init__(self, sentences_per_chunk: int = 3):
            # Initialize the chunker with the number of sentences per chunk
            self.sentences_per_chunk = sentences_per_chunk

        def split_text(self, text: str) -> list[str]:
            # Handle the case where the input text is empty
            if not text:
                return []

            # Split the input text into sentences using regular expression
            # Regex looks for white space following . ! or ? and makes a split
            sentences = re.split(r"(?<=[.!?])\s+", text)
            chunks = []

            # Group sentences into chunks based on the specified number
            for i in range(0, len(sentences), self.sentences_per_chunk):
                # Combine sentences into a single chunk
                chunk = " ".join(sentences[i : i + self.sentences_per_chunk])
                chunks.append(chunk)

            # Return the list of chunks
            return chunks

    return (SentenceChunker,)


@app.cell
def _(SentenceChunker, embedding_function, general_set):
    sentence_chunker = SentenceChunker(sentences_per_chunk=10)

    sent_collection, sent_questions_collection = general_set.get_collections(
        sentence_chunker,
        embedding_function,
    )
    return sent_collection, sent_questions_collection


@app.cell
def _(Evaluation, sent_collection, sent_questions_collection):
    _eval = Evaluation(
        "datasets/general_evaluation/questions_df.csv",
        sent_collection,
        sent_questions_collection,
    )

    _results = _eval.evaluate(retrieve=5)

    print_metrics(_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### RecursiveTokenChunker""")
    return


@app.cell
def _(RecursiveTokenChunker, embedding_function, general_set):
    chunker = RecursiveTokenChunker(chunk_size=1000, chunk_overlap=200)


    collection, questions_collection = general_set.get_collections(
        chunker,
        embedding_function,
        # db_to_save_chunks="chroma_db/general_chunks_db/",
        # db_to_save_questions="chroma_db/general_questions_db/",
    )
    return collection, questions_collection


@app.cell
def _(Evaluation, collection, questions_collection):
    _eval = Evaluation(
        "datasets/general_evaluation/questions_df.csv",
        collection,
        questions_collection,
    )

    _results = _eval.evaluate(retrieve=5)

    print_metrics(_results)
    return


@app.cell
def _(mo):
    mo.md(r"""###  Synthetic Data Evaluation""")
    return


@app.cell
def _(Path):
    import os
    from dotenv import load_dotenv

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    corpora_dir_path = Path('datasets/raydocs_full')

    corpora_paths = [
        path
        for path in corpora_dir_path.rglob('*')
        if path.is_file()
    ]

    queries_csv_path = 'datasets/raydocs-generated-queries-and-excerpts.csv'
    return corpora_paths, openai_api_key, queries_csv_path


@app.cell
def _(corpora_paths, openai_api_key, queries_csv_path):
    from evaluation import SyntheticEvalSet

    synth_set = SyntheticEvalSet(
        corpora_paths,
        queries_csv_path,
        openai_api_key=openai_api_key,
        prompt_path="datasets/prompts"
    )

    synth_set.generate_queries_and_excerpts(
        approximate_excerpts=True, 
        num_rounds=1,
        queries_per_corpus=3,
    )
    return


if __name__ == "__main__":
    app.run()
