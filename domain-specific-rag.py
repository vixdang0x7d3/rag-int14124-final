import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo

    import os

    import torch
    from pathlib import Path
    from dotenv import load_dotenv

    import chromadb
    from chromadb.utils import embedding_functions

    from utils import (
        openai_token_count, 
        rigorous_document_search,
    )

    from chunker import BaseChunker, RecursiveTokenChunker


@app.cell(hide_code=True)
def _():
    mo.md("""# Machine Specs""")
    return


@app.cell
def _():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        mo.output.append("CUDA is available")
        mo.output.append(f"GPU: {torch.cuda.get_device_name(0)}")
    return device, hf_token


@app.cell(hide_code=True)
def _():
    mo.md("""### Ray docs Pre-processing""")
    return


@app.cell
def _():

    raydocs_dir_path = Path("datasets/raydocs_full")

    raydocs_paths = [path for path in raydocs_dir_path.rglob("*") if path.is_file()]

    return (raydocs_paths,)


@app.cell
def _(device, hf_token):

    # Define chunking algorithm (best setting)
    recursive_chunker = RecursiveTokenChunker(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=openai_token_count,
    )

    # Load finetuned embedding model
    finetuned_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="thanhpham1/Fine-tune-all-mpnet-base-v2",
        device=device,
        token=hf_token,
    )

    return (recursive_chunker,)


@app.function
def extract_chunks_and_metadata(splitter: BaseChunker, doc_list: list[str]):
    """
    Accept a chunking algorithm, use it to split text
    then compute metadata for each chunk.
    """

    chunks = []
    metadatas = []

    for doc_id in doc_list:
        doc_path = doc_id

        import platform

        if platform.system() == "Windows":
            with open(doc_path, "r", encoding="utf-8") as f:
                doc = f.read()
        else:
            with open(doc_path, "r") as f:
                doc = f.read()

        current_chunks = splitter.split_text(doc)
        current_metadatas = []

        for chunk in current_chunks:
            start_index = end_index = -1
            try:
                _, start_index, end_index = rigorous_document_search(
                    doc,
                    chunk,
                )  # type: ignore
            except Exception as e:
                print(f"Error in finding '{chunk}': {e}")

            current_metadatas.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "corpus_id": doc_id,
                }
            )

        chunks.extend(current_chunks)
        metadatas.extend(current_metadatas)
    return chunks, metadatas


@app.cell
def _(raydocs_paths, recursive_chunker):
    chunks, metadatas = extract_chunks_and_metadata(
        recursive_chunker, doc_list=raydocs_paths
    )
    return


app._unparsable_cell(
    r"""
    def chunks_and_metadatas_to_collection(
        chunks,
        metadatas,
        embedding_function,
        collection_name: str | None = None,
        save_path: str | None = None,
    ):
        collection = None
        if save_path is not None and collection_name is not None:
            try:
                chunk_client = chromadb.PersistentClient()
            except Exception as e:
    
    """,
    name="*chunker_to_collection"
)


if __name__ == "__main__":
    app.run()
