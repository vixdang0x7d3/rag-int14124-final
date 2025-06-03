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


@app.cell
def _():
    import torch

    device = "cuda" if torch.cuda.is_available else "cpu"
    print(f"Using CUDA ? {'YES' if device == 'cuda' else 'NO'}")
    if device:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Testing code""")
    return


@app.cell
def _(device):
    from chromadb.utils import embedding_functions
    from evaluation import GeneralEvalSet
    from chunker import RecursiveTokenChunker

    from pathlib import Path

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2",
        device=device,
    )

    chunker = RecursiveTokenChunker()

    general_set = GeneralEvalSet(
        general_benchmark_path=Path("datasets/general_evaluation")
    )
    return chunker, embedding_function, general_set


@app.cell
def _(chunker, embedding_function, general_set):
    collection, question_collection = general_set.get_collections(
        chunker,
        embedding_function,
        db_to_save_chunks="chroma_db/general_chunks_db/",
        db_to_save_questions="chroma_db/general_questions_db/",
    )
    return


if __name__ == "__main__":
    app.run()
