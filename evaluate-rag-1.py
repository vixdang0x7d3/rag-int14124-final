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
        Evaluation,
        GeneralEvalSet,
        Path,
        RecursiveTokenChunker,
        embedding_functions,
        pd,
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
    return


@app.cell
def _(GeneralEvalSet, Path, RecursiveTokenChunker, embedding_function):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _chunkers = [
        RecursiveTokenChunker(
            chunk_size=1000,
            chunk_overlap=200,
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

    def process_chunker(chunker, embedding_function):
        general_set = GeneralEvalSet(Path("datasets/general_evaluation"))
        c, q = general_set.get_collections(chunker, embedding_function)

    collection_pairs = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_chunker, ch, embedding_function) for ch in _chunkers
        ]
        for future in as_completed(futures):
            collection_pairs.append(future.result())

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
