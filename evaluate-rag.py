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

    device = 'cuda' if torch.cuda.is_available else 'cpu'
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
    return (embedding_functions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Naive chunking stratergy

    We'll use the `BaseChunker` class to define our own. At it's core `BaseChunker` is very simple:

    ```python
    class BaseChunker(ABC):
        @abstractmethod
        def split_text(self, text: str) -> list[str]:
            pass
    ```

    It expects only a `split_text` method that can take in a string and return a list of strings, which is our chunks. The transformation along the way can be more creatively defined.

    We'll use this as a base to reimplement our naive chunker.
    """
    )
    return


@app.cell
def _(BaseChunker):
    from spacy.lang.en import English


    class SentenceChunker(BaseChunker):
        def __init__(self, sentences_per_chunk: int = 3):
            self.sentences_per_chunk = sentences_per_chunk
            self.nlp = English()
            self.nlp.add_pipe("sentencizer")

        def split_text(self, text: str) -> list[str]:
            chunk_size = self.sentences_per_chunk

            if not text:
                return []

            sentences = list(self.nlp(text).sents)
            sentences = [str(sent) for sent in sentences]

            chunks = []

            for i in range(0, len(sentences), chunk_size):
                chunk = ' '.join(sentences[i:i+chunk_size])
                chunks.append(chunk)

            return chunks
    return (SentenceChunker,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Defining a embedding function and chunker""")
    return


@app.cell
def _(SentenceChunker, device, embedding_functions):
    sent_trans_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='all-mpnet-base-v2',
        device=device
    )

    sentence_chunker = SentenceChunker(sentences_per_chunk = 10)

    sent_trans_ef.__class__.__name__
    return sent_trans_ef, sentence_chunker


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Start General Evaluation""")
    return


@app.cell
def _(GeneralEvaluation, sent_trans_ef, sentence_chunker):

    evaluation = GeneralEvaluation()


    results = evaluation.run(
        sentence_chunker, 
        sent_trans_ef,
        db_to_save_chunks="datasets/general_evaluation/naive-configuration"
    )

    return


if __name__ == "__main__":
    app.run()
