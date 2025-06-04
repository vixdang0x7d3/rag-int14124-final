import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Install the ChromaDB Evaluation Framework package

    For `pip` users:


    ```
    pip add git+https://github.com/brandonstarxel/chunking_evaluation.git
    ```
    """
    )
    return


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
        pd,
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


app._unparsable_cell(
    r"""
    import platform

    print(platform.system()
    """,
    name="_"
)


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
    chunker = RecursiveTokenChunker()

    collection, questions_collection = general_set.get_collections(
        chunker,
        embedding_function,
        db_to_save_chunks="chroma_db/general_chunks_db/",
        db_to_save_questions="chroma_db/general_questions_db/",
    )
    return chunker, collection, questions_collection


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
def _(GeneralEvalSet, Path, RecursiveTokenChunker, embedding_function):
    _chunkers = [
        RecursiveTokenChunker(
            chunk_size=800,
            chunk_overlap=400,
        ),
        RecursiveTokenChunker(
            chunk_size=800,
            chunk_overlap=400,
        ),
        RecursiveTokenChunker(
            chunk_size=400,
            chunk_overlap=200,
        ),
        RecursiveTokenChunker(
            chunk_size=200,
            chunk_overlap=0,
        ),
    ]

    collection_pairs = []

    for _chunker in _chunkers:
        _general_set = GeneralEvalSet(
            Path("datasets/general_evaluation"),
        )
        c, q = _general_set.get_collections(
            _chunker,
            embedding_function,
        )

        collection_pairs.append((c, q))

    return (collection_pairs,)


@app.cell
def _(
    Evaluation,
    chunk_overlap,
    chunk_size,
    chunker,
    collection_pairs,
    pd,
    result,
    results,
):
    _results = []
    _df = pd.DataFrame()

    for c_collection, q_collection in collection_pairs:
        eval = Evaluation(
            "datasets/general_evaluation/questions_df.csv",
            c_collection,
            q_collection,
        )
        _result = eval.evaluate(retrieve=5)

        del _result["corpora_scores"]  # Remove detailed scores for brevity

        _chunk_size = chunker._chunk_size if hasattr(chunker, "_chunk_size") else 0
        _chunk_overlap = (
            chunker._chunk_overlap if hasattr(chunker, "_chunk_overlap") else 0
        )
        _result["chunker"] = (
            chunker.__class__.__name__ + f"_{chunk_size}_{chunk_overlap}"
        )

        _results.append(result)

    df = pd.DataFrame(results)
    return


if __name__ == "__main__":
    app.run()
