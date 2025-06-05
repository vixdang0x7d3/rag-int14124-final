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
    import os
    import re


    import torch
    import pandas as pd

    from dotenv import load_dotenv

    from evaluation import SyntheticEvalSet, Evaluation
    from chunker import BaseChunker, RecursiveTokenChunker

    from chromadb.utils import embedding_functions

    from pathlib import Path

    return (
        Evaluation,
        Path,
        RecursiveTokenChunker,
        SyntheticEvalSet,
        embedding_functions,
        load_dotenv,
        os,
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
def _(Path, SyntheticEvalSet, load_dotenv, os):
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    corpora_dir_path = Path("datasets/raydocs_full")
    corpora_paths = [path for path in corpora_dir_path.rglob("*") if path.is_file()]
    queries_csv_path = "datasets/raydocs-generated-queries-and-excerpts.csv"

    synth_set = SyntheticEvalSet(
        corpora_paths, 
        queries_csv_path, 
        openai_api_key=openai_api_key,
        prompt_path="datasets/prompts",
    )
    return queries_csv_path, synth_set


@app.cell
def _(RecursiveTokenChunker, embedding_function, synth_set):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _chunkers = [
        RecursiveTokenChunker(
            chunk_size=1200,
            chunk_overlap=100,
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

    def process_chunker(synth_set, chunker, embedding_function):
        c, q = synth_set.get_collections(chunker, embedding_function)
        return (c,q)

    collection_pairs = []
    for ch in _chunkers:
            c, q = process_chunker(synth_set, ch, embedding_function)
            collection_pairs.append((c, q))

    return (collection_pairs,)


@app.cell
def _(collection_pairs):
    collection_pairs
    return


@app.cell
def _(
    Evaluation,
    chunk_overlap,
    chunk_size,
    chunker,
    collection_pairs,
    pd,
    queries_csv_path,
    result,
    results,
):
    _results = []
    _df = pd.DataFrame()

    for c_collection, q_collection in collection_pairs:

        print(c_collection)
        print(q_collection)
    
        _eval = Evaluation(
            queries_csv_path,
            c_collection,
            q_collection,
        )
        _result = _eval.evaluate(retrieve=5)

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

    df
    return


if __name__ == "__main__":
    app.run()
