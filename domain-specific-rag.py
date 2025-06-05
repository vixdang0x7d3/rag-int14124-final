import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo

    import os
    import random

    import torch
    from pathlib import Path
    from dotenv import load_dotenv

    from sentence_transformers import SentenceTransformer

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

    raydocs_paths = [str(path) for path in raydocs_dir_path.rglob("*") if path.is_file()]

    return (raydocs_paths,)


@app.cell
def _(device, hf_token):
    # Define chunking algorithm (best setting)
    recursive_chunker = RecursiveTokenChunker(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=openai_token_count,
    )

    # Load finetuned embedding model
    finetuned_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="thanhpham1/Fine-tune-all-mpnet-base-v2",
        device=device,
        token=hf_token,
    )

    finetuned_em = SentenceTransformer(
        "thanhpham1/Fine-tune-all-mpnet-base-v2",
        device=device,
        token=hf_token,
    )

    return finetuned_ef, finetuned_em, recursive_chunker


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
    return chunks, metadatas


@app.cell
def _(metadatas):
    metadatas
    return


@app.cell
def _():
    mo.md(r"""### Create vector embeddings, save to database""")
    return


@app.function
def chunks_and_metadata_to_collection(
    chunks,
    metadatas,
    embedding_function,
    collection_name: str | None = None,
    save_path: str | None = None,
):
    collection = None
    if save_path is not None and collection_name is not None:
            chunk_client = chromadb.PersistentClient()
            collection = chunk_client.create_collection(
                collection_name, 
                embedding_function=embedding_function,
                metadata={"hnsw:search_ef":50}
            )
            chunk_client.persist()

    if collection is None:
        inmem_client = chromadb.EphemeralClient()
        try:
            inmem_client.delete_collection("inmem-document-index")
        except Exception:
            pass

        collection = inmem_client.create_collection(
            "inmem-document-index",
            embedding_function=embedding_function,
            metadata={"hnsw:search_ef":50}
        )

    BATCH_SIZE = 500
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i+BATCH_SIZE]
        batch_metas = metadatas[i:i+BATCH_SIZE]
        batch_ids = [str(j) for j in range(i, i+len(batch_chunks))]
        collection.add(
            documents=batch_chunks,
            metadatas=batch_metas,
            ids=batch_ids
        )

    return collection


@app.cell
def _(chunks, finetuned_ef, metadatas):
    raydocs_collection = chunks_and_metadata_to_collection(
        chunks,
        metadatas, 
        finetuned_ef,
    )
    return (raydocs_collection,)


@app.cell
def _(raydocs_collection):
    raydocs_collection
    return


@app.cell
def _():
    mo.md(r"""### Semantic search on vector embeddingg.""")
    return


@app.function
def retrieve_documents(
    collection: chromadb.Collection,
    emb_model: SentenceTransformer,
    queries: list[str], 
    n_results: int = 5
) -> list[list[str]]:
    query_embeddings = emb_model.encode(queries, batch_size=100).tolist()
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results
    )

    return results.get("documents", [[] for _ in queries])


@app.cell
def _():
    sample_queries = [
            "What is ray ?",
            "What is the difference between ray and spark?",
            "How to install ray?"
    ]
    sample_queries
    return (sample_queries,)


@app.cell
def _(finetuned_em, raydocs_collection):
    retrieved = retrieve_documents(
        raydocs_collection,
        finetuned_em,
        queries=[
            "What is ray ?",
            "What is the difference between ray and spark?",
            "How to install ray?"
        ]
    )

    retrieved
    return (retrieved,)


app._unparsable_cell(
    r"""
    import textwrap
    def print_wrapped(text, wrap_lenght=80):
        wrapped_text = textwrap.fill(text, wrap_lenght)
        print(wrapped_text)

    def print_wrapped_output(text, wrap_length: 80):
        \"\"\"
            New print_wrapped version that respect the
            indentations of the LLM output and the prompt
            \"\"\"
            for line in text.splitlines():
                indent = len(line) - len(line.lstrip())
                wrapped = textwrap.fill(
                    line, 
                    width=wrap_length, 
                    subsequent_indent=' ' * indent, 
                    replace_whitespace=False, 
                    drop_whitespace=False
                )
                print(wrapped)
    """,
    name="_"
)


@app.cell
def _(print_wrapped, retrieved, sample_queries):
    print(f"Query: {sample_queries[0]}")
    print("\nSemantically similar texts:\n")

    for text in retrieved[0]:
        print("Text:")
        print_wrapped(text)
        print("\n")
    return


@app.cell
def _():
    mo.md(r"""### LLM for local generation""")
    return


@app.cell
def _():
    mo.md(r"""### Checking local GPU memory availability""")
    return


@app.cell
def _():
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2**30))
    print(f"Available GPU memory: {gpu_memory_gb} GB")
    return (gpu_memory_gb,)


@app.cell
def _():
    mo.md(r"""### Adaptive quantization configuration based on current hardware""")
    return


@app.cell
def _(gpu_memory_gb):
    if gpu_memory_gb < 5.0:
        print(f"Your available GPU memory is {gpu_memory_gb}GB. You may not have enough memory to run a LLaMA model locally without heavy quantization.")
        # Consider using LLaMA 2 7B or 3 8B in 4-bit quantized formats like Q4_0 or Q4_K_M
        use_quantization_config = True
        model_id = "meta-llama/Llama-2-7b-chat-hf"  # Example; replace with actual quantized version
    elif gpu_memory_gb < 8.1:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: LLaMA 2 7B or LLaMA 3 8B in 4-bit quantization.")
        use_quantization_config = True
        model_id = "meta-llama/Llama-2-7b-chat-hf"
    elif gpu_memory_gb < 16.0:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: LLaMA 3 8B in float16 or LLaMA 2 13B in 4-bit precision.")
        use_quantization_config = False
        model_id = "meta-llama/Llama-3-8b-chat-hf"
    elif gpu_memory_gb < 30.0:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: LLaMA 3 70B or LLaMA 2 70B in 4-bit quantization.")
        use_quantization_config = True
        model_id = "meta-llama/Llama-3-70b-chat-hf"
    else:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: LLaMA 3 70B or LLaMA 2 70B in float16 precision.")
        use_quantization_config = False
        model_id = "meta-llama/Llama-3-70b-chat-hf"

    print(f"use_quantization_config set to: {use_quantization_config}")
    print(f"model_id set to: {model_id}")

    return model_id, use_quantization_config


@app.cell
def _():
    mo.md(r"""### Loading an LLM locally""")
    return


@app.cell
def _(model_id, use_quantization_config):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import is_flash_attn_2_available
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
      attn_implementation = "flash_attention_2"
    else:
      attn_implementation = "sdpa"
    
    print(f"[INFO] Using attention implementation: {attn_implementation}")
    print(f"[INFO] Using model_id: {model_id}")
    print(f"[INFO] Using quantization config?: {'YES' if use_quantization_config else 'NO'}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_id
    )

    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config if use_quantization_config else None,
        low_cpu_mem_usage=False,
        attn_implementation=attn_implementation,
    )

    if not use_quantization_config:
        llm_model.to("cuda")
    return AutoTokenizer, llm_model, tokenizer


@app.cell
def _(AutoTokenizer):
    def prompt_builder(
        query: str,
        context: list[str],
        tokenizer: AutoTokenizer,
    ) -> str:
        """
        Augments query with context using LLaMA-style prompt (no apply_chat_templates).
        """

        # Format the context as bullet points
        context_text = '- ' + '\n- '.join(context)

        # Create the system prompt
        system_prompt = (
            "Based on the following context items, please answer the query.\n"
            "Make sure your answers are as explanatory as possible.\n"
            "If you don't know the answer just return <NONE>, don't make things up"
        )

        # Format final prompt string
        prompt = (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"Now use the following context items to answer the user query:\n"
            f"{context_text}\n\n"
            f"Relevant passages: <extract relevant passages from the context here>\n"
            f"User query: {query}\n"
            f"Answer: [/INST]"
        )

        prompt.format(system_prompt=system_prompt, context_text=context_text, query=query)
        return prompt
    return (prompt_builder,)


@app.cell
def _():
    gpt4_questions = [
       "What is Ray, and what are it's primary use cases?",
       "How does Ray simplify distributed computing for Python applications?"
       "What are Ray's AI libraries, and what functionalities do they offer?"
       "What is the role of the Ray Dashboard, and how can it be accessed?" 
    ]

    # questions pulled from faq pages
    manual_questions = [ 
        "How can i develop and test Tune locally",
        "How can i use Tune with Docker",
        "Do Ray clusters support multi-tenancy"
    ]

    query_list = gpt4_questions + manual_questions
    return (query_list,)


@app.cell
def _(finetuned_em, prompt_builder, query_list, raydocs_collection, tokenizer):
    _query = random.choice(query_list)
    print(f"Query: {_query}\n")


    context_items = retrieve_documents(
        raydocs_collection,
        finetuned_em,
        queries=[_query]
    )[0]

    prompt = prompt_builder(
        query=_query,
        context=context_items,
        tokenizer=tokenizer
    )

    print(prompt)
    return (prompt,)


@app.cell
def _(llm_model, prompt, query, tokenizer):
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=0.7, # lower temperature = more deterministic outputs, higher temperature = more creative outputs
                                 do_sample=True, # whether or not to use sampling, see https://huyenchip.com/2024/01/16/sampling.html for more
                                 max_new_tokens=256) # how many new tokens to generate from prompt 

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    print(f"Query: {query}")
    print(f"RAG answer:\n{output_text.replace(prompt, '')}")
    return


@app.function
def run_rag_pipeline(
    query: str,
    retrieve_fn,
    collection,
    embedding_function,
    tokenizer,
    llm_model,
    prompt_builder_fn,
    device: str = "cuda",
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run a full Retrieval-Augmented Generation (RAG) pipeline for a single query.

    Args:
        query: A single user query string.
        retrieve_fn: Function to retrieve relevant context documents.
        collection: Vector database or document store.
        embedding_function: Embedding model used for retrieval.
        tokenizer: Tokenizer matching the LLM.
        llm_model: The language model (must support `.generate()`).
        prompt_builder_fn: Function to build the augmented prompt.
        device: Device to run model inference on ("cuda" or "cpu").
        temperature: Sampling temperature for text generation.
        max_new_tokens: Maximum number of tokens to generate.
        do_sample: Whether to sample during generation.
        verbose: Whether to print intermediate steps.

    Returns:
        dict with keys: query, context, prompt, output_text, answer
    """

    if verbose:
        print(f"\nQuery: {query}\n")

    # Retrieve context items
    context_items = retrieve_fn(
        collection,
        embedding_function,
        queries=[query]
    )[0]

    # Build prompt
    prompt = prompt_builder_fn(
        query=query,
        context=context_items,
        tokenizer=tokenizer
    )

    if verbose:
        print(f"\nPrompt:\n{prompt}\n")

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate response
    outputs = llm_model.generate(
        **input_ids,
        temperature=temperature,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens
    )

    # Decode and clean output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = output_text.replace(prompt, "").strip()

    if verbose:
        print(f"RAG Answer:\n{answer}\n")

    return {
        "query": query,
        "context": context_items,
        "prompt": prompt,
        "output_text": output_text,
        "answer": answer,
    }


if __name__ == "__main__":
    app.run()
