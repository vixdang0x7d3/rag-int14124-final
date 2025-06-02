from chromadb.utils import embedding_functions
import marimo
from torch import device

from evaluation.base_evaluation import BaseDatasetCollection

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
    mo.md(r"""### Import and dependencies""")
    return


@app.cell
def _():
    import os
    import pandas as pd
    from chromadb.utils import embedding_functions

    from evaluation import BaseDatasetCollection

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Setup codes that will be GeneralDatasetCollection""")
    return


@app.cell
def _(mo):
    from pathlib import Path

    # These are parameters
    chroma_db_path = Path("chroma_db/general_evaluation")
    general_dataset_path = Path("datasets/general_evaluation")

    questions_df_path = general_dataset_path / "questions_df.csv"

    corpora_dir_path = general_dataset_path / "corpora"
    corpora_filenames = [f for f in corpora_dir_path.iterdir() if f.is_file()]

    corpora_id_paths = {f.stem: str(f) for f in corpora_filenames}

    mo.output.append(corpora_id_paths)
    mo.output.append(questions_df_path)

    return (questions_df_path, corpora_id_paths, chroma_db_path)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Testing code""")
    return


@app.cell
def _(questions_df_path, corpora_id_paths, chroma_db_path, device):
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2",
        device=device,
    )

    chunker = None

    base = BaseDatasetCollection(
        str(questions_df_path),
        str(chroma_db_path),
        corpora_id_paths,
    )

    collection, questions_collection = base.get_collections(
        embedding_function,
        chunker,
    )

    return


if __name__ == "__main__":
    app.run()
