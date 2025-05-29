import marimo

__generated_with = "0.13.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    return mo, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # What we're going to build?

    A simple RAG pipeline that's able to process a PDF document - [nutrition-textbook](https://pressbooks.oer.hawaii.edu/humannutrition2/),

    We'll write the code to:

    1. Open a PDF document & extract the text.
    2. Format the text into appropriate chunks for feeeding them into an embedding model.
    3. Embed the text aka. turn them into numerical representation which we can store for later use.
    4. Build a **retrieval system** that finds relevant chunks of text based on a query
    5. Create a prompt that incorporates the retrieved pieces of text.
    6. Generate an answer to the query based on texts from the textbook.
    """
    )
    return


@app.cell
def _():

    # if "COLAB_GPU" in os.environ:
    #     print("[INFO] Running in Google Colab, installing requirements.")
    #     !pip install torch==2.6.0 # requires torch 2.1.1+ (for efficient sdpa implementation)
    #     !pip install PyMuPDF # for reading PDFs with Python
    #     !pip install tqdm # for progress bars
    #     !pip install sentence-transformers # for embedding models
    #     !pip install accelerate # for quantization model loading
    #     !pip install bitsandbytes # for quantizing models (less storage space)
    #     !pip install flash-attn --no-build-isolation # for faster attention mechanism = faster LLM inference
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Document/Text Processing and Embedding Generation""")
    return


@app.cell
def _(os):
    import requests

    from pathlib import Path

    pdf_path = Path(os.getcwd()) / "data" / "human-nutrition-text.pdf"

    if not pdf_path.parent.exists():
        print(f"Creating directory {pdf_path.parent}")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        print("File not found, downloading...")

        url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

        filename = pdf_path.name

        response = requests.get(url)

        if response.status_code == 200:
            pdf_path.write_bytes(response.content)
            print(f"Downloaded {filename} to {pdf_path}")
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
    else:
        print(f"File already exists at {pdf_path}.")
    return Path, pdf_path


@app.cell
def _(Path, pdf_path):
    import pymupdf
    from tqdm import tqdm

    def clean_text(text: str) -> str:
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text

    def extract_text(pdf_path: Path) -> list[dict]:
        with pymupdf.open(pdf_path) as doc:
            text_data = []

            for pageno, page in tqdm(enumerate(doc), desc="Extracting text", total=len(doc)):
                text = page.get_text("text")
                text = clean_text(text)

                # Content start after page 42
                # 1 token = 4 characters
                text_data.append({
                    "page_number": pageno - 42,
                    "page_char_count" : len(text),
                    "page_word_count" : len(text.split()),
                    "page_sentence_count_raw" : len(text.split(".")),
                    "page_token_count" : len(text) / 4,
                    "text": text
                })

            return text_data

    text_data = extract_text(pdf_path)
    text_data[:2]
    return text_data, tqdm


@app.cell
def _(text_data):
    from pprint import pprint
    import random

    pprint(
        random.sample(text_data, k=3)
    )
    return pprint, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Get some stat on the text""")
    return


@app.cell
def _(text_data):
    import pandas as pd

    df = pd.DataFrame(text_data)
    df.head()
    return df, pd


@app.cell
def _(df):
    df.describe().round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Splitting pages into sentences""")
    return


@app.cell
def _():
    from spacy.lang.en import English

    nlp = English()

    nlp.add_pipe("sentencizer")

    doc = nlp("This is a sentence. This is anothere sentence")
    assert len(list(doc.sents)) == 2

    list(doc.sents)
    return (nlp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Perform transformation on our text""")
    return


@app.cell
def _(nlp, text_data, tqdm):
    for item in tqdm(text_data):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sent) for sent in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return


@app.cell
def _(random, text_data):
    random.sample(text_data, k=1)
    return


@app.cell
def _(pd, text_data):
    df_1 = pd.DataFrame(text_data)
    df_1.describe().round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create chunks from sentences""")
    return


@app.cell
def _(text_data, tqdm):
    sent_per_chunk = 10

    def split_list(input_list: list[str], chunk_size: int) -> list[list[str]]:
        """Split list of sentences into chunk lenght sub-lists"""
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
    for item_1 in tqdm(text_data):
        item_1['chunks'] = split_list(item_1['sentences'], chunk_size=sent_per_chunk)
        item_1['num_chunks'] = len(item_1['chunks'])
    return


@app.cell
def _(random, text_data):
    random.sample(text_data, k=1)
    return


@app.cell
def _(pd, text_data):
    df_2 = pd.DataFrame(text_data)
    df_2.describe().round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Merging chunks from list to a single string""")
    return


@app.cell
def _(text_data, tqdm):
    import re
    chunk_data = []
    for item_2 in tqdm(text_data):
        for chunk in item_2['chunks']:
            chunk_dict = {}
            chunk_dict['page_number'] = item_2['page_number']
            merged_chunk = ''.join(chunk).replace('  ', ' ').strip()
            merged_chunk = re.sub('\\.([A-Z])', '. \\1', merged_chunk)
            chunk_dict['chunk'] = merged_chunk
            chunk_dict['chunk_char_count'] = len(merged_chunk)
            chunk_dict['chunk_word_count'] = len([word for word in merged_chunk.split(' ')])
            chunk_dict['chunk_token_count'] = len(merged_chunk) / 4
            chunk_data.append(chunk_dict)
    len(chunk_data)
    return (chunk_data,)


@app.cell
def _(chunk_data, random):
    random.sample(chunk_data, 1)
    return


@app.cell
def _(chunk_data, pd):
    df_3 = pd.DataFrame(chunk_data)
    df_3.describe().round(2)
    return (df_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Some of the chunks have quite low token count. We will filter out samples with less than 30 tokens and see if they are worth keeping""")
    return


@app.cell
def _(df_3):
    min_token_length = 30
    for row in df_3[df_3['chunk_token_count'] <= min_token_length].sample(5).iterrows():
        print(f"Token count: {row[1]['chunk_token_count']}\nText: {row[1]['chunk']}\n\n")
    return (min_token_length,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Many of these are page headers and footers, they don't seem to offer much information. \
    We can remove them and keep only chunk dicts with over 30 tokens.
    """
    )
    return


@app.cell
def _(df_3, min_token_length, pprint):
    chunk_data_1 = df_3[df_3['chunk_token_count'] > min_token_length].to_dict(orient='records')
    pprint(chunk_data_1[:2])
    return (chunk_data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Embedding text chunks""")
    return


@app.cell
def _():
    from sentence_transformers import SentenceTransformer, util

    # TODO(vi): Research SentenceTransformer

    embedding_model = SentenceTransformer(
        model_name_or_path="all-mpnet-base-v2",
    )

    sentences = [
        "The Sentences Transformers library provides an easy and open-source way to create embeddings.",
        "Sentences can be embedded one by one or as a list of strings.",
        "Embeddings are one of the most powerful concepts in machine learning!",
        "Learn to use embeddings well and you'll be well on your way to being an AI engineer."
    ]

    embeddings = embedding_model.encode(sentences)

    embedding_dict = dict(zip(sentences, embeddings))

    for sent, emb, in embedding_dict.items():
        print(f"Sentence: {sent}\n")
        print(f"Embedding: {emb}\n")
        print("-----------------------------------\n\n")
    return SentenceTransformer, embedding_model, sentences, util


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Using a GPU can significantly speed up this step""")
    return


@app.cell
def _():
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using CUDA? {'YES' if device=='cuda' else 'NO'}")
    if device:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device, torch


@app.cell
def _(device, embedding_model, sentences):
    embedding_model.to(device)
    _ = embedding_model.encode(sentences)
    return


@app.cell
def _(chunk_data_1, embedding_model):
    text_chunks = [item['chunk'] for item in chunk_data_1]
    embeddings_1 = embedding_model.encode(text_chunks, batch_size=32)
    return (embeddings_1,)


@app.cell
def _(chunk_data_1, embeddings_1):
    for chunk_1, emb_1 in zip(chunk_data_1, embeddings_1):
        chunk_1['embedding'] = emb_1
    chunk_data_1[0]
    return


@app.cell
def _(chunk_data_1, pd):
    chunk_embedding_df = pd.DataFrame(chunk_data_1)
    save_path = 'data/chunk_embedding_df.parquet'
    chunk_embedding_df.to_parquet(save_path, index=False)
    return (save_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Try loading them up""")
    return


@app.cell
def _(pd, save_path):
    chunk_embedding_df_1 = pd.read_parquet(save_path)
    chunk_embedding_df_1.head()
    return (chunk_embedding_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Storing as CSV requires objects get serialized into strings so we used parquet format instead""")
    return


@app.cell
def _(chunk_embedding_df_1):
    chunk_embedding_df_1.loc[0, 'embedding']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Retrieval

    At the moment, we have our document index ready in the form of a simple dataframe. \
    In this stage, we'll convert our embedding into tensor for GPU accelerated computation and define a similarity search function that can retrieve $k$ relevant text passages based on a user query
    """
    )
    return


@app.cell
def _(chunk_embedding_df_1, torch):
    import numpy as np
    device_1 = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings_2 = torch.tensor(np.array(chunk_embedding_df_1['embedding'].tolist()), dtype=torch.float32).to(device_1)
    embeddings_2.shape
    return device_1, embeddings_2


@app.cell
def _(embeddings_2):
    embeddings_2[0]
    return


@app.cell
def _(SentenceTransformer, device_1):
    embedding_model_1 = SentenceTransformer(model_name_or_path='all-mpnet-base-v2', device=device_1)

    embedding_model_1
    return (embedding_model_1,)


@app.cell
def _(embedding_model_1, embeddings_2, torch, util):
    query = 'macronutrients functions'
    query_embedding = embedding_model_1.encode(query, convert_to_tensor=True)
    from time import perf_counter as timer
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings_2)[0]
    end_time = timer()
    print(f'Query: {query}')
    print(f'Time taken to get scores on {len(embeddings_2)} embeddings: {end_time - start_time:.5f} seconds.')
    top_5 = torch.topk(dot_scores, k=5)
    top_5
    return query, top_5


@app.cell
def _():
    import textwrap

    def print_wrapped(text, wrap_length=80):
        wrapped_text = textwrap.fill(text, wrap_length)
        print(wrapped_text)
    return print_wrapped, textwrap


@app.cell
def _(chunk_data_1, print_wrapped, query, top_5):
    print(f"Query: '{query}'")
    print('Result:')
    for score, idx in zip(top_5[0], top_5[1]):
        print(f'Score: {score:.4f}')
        print('Text: ')
        print_wrapped(chunk_data_1[idx]['chunk'])
        print(f"Page No. : {chunk_data_1[idx]['page_number']}\n\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Define similarity functions""")
    return


@app.cell
def _(torch):
    def dot_product(vec_1, vec_2):
        return torch.dot(vec_1, vec_2)


    def cosine_similarity(vec_1, vec_2):
        dot_product = torch.dot(vec_1, vec_2)

        norm_vec_1 = torch.sqrt(torch.sum(vec_1 ** 2))
        norm_vec_2 = torch.sqrt(torch.sum(vec_2 ** 2))

        return dot_product / (norm_vec_1 * norm_vec_2)


    # Example tensors
    vector1 = torch.tensor([1, 2, 3], dtype=torch.float32)
    vector2 = torch.tensor([1, 2, 3], dtype=torch.float32)
    vector3 = torch.tensor([4, 5, 6], dtype=torch.float32)
    vector4 = torch.tensor([-1, -2, -3], dtype=torch.float32)

    # Calculate dot product
    print("Dot product between vector1 and vector2:", dot_product(vector1, vector2))
    print("Dot product between vector1 and vector3:", dot_product(vector1, vector3))
    print("Dot product between vector1 and vector4:", dot_product(vector1, vector4))

    # Calculate cosine similarity
    print(
        "Cosine similarity between vector1 and vector2:",
        cosine_similarity(vector1, vector2),
    )
    print(
        "Cosine similarity between vector1 and vector3:",
        cosine_similarity(vector1, vector3),
    )
    print(
        "Cosine similarity between vector1 and vector4:",
        cosine_similarity(vector1, vector4),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Functionizing the semantic search pipeline""")
    return


@app.cell
def _(SentenceTransformer, embeddings_2, print_wrapped, torch, util):
    def retrieve_chunks(
        query: str, emdeddings: torch.tensor, 
        model: SentenceTransformer, 
        topk: int=5
    ):
        query_embedding = model.encode(query, convert_to_tensor=True)
        dot_scores = util.dot_score(query_embedding, embeddings_2)[0]
        scores, indices = torch.topk(dot_scores, k=topk)
        return (scores, indices)

    def print_topk(
        query: str, 
        embeddings: torch.tensor, 
        document_index: list[dict], 
        model: SentenceTransformer, 
        topk: int=5
    ):
        scores, indices = retrieve_chunks(query, embeddings, model)
        print(f'Query: {query}')
        print('Result: ')
        for score, idx in zip(scores, indices):
            print(f'Score: {score:.4f}')
            print_wrapped(document_index[idx]['chunk'])
            print(f"Page No. : {document_index[idx]['page_number']}\n\n")
    return print_topk, retrieve_chunks


@app.cell
def _(embedding_model_1, embeddings_2, retrieve_chunks):
    query_1 = 'symtomps of pellagra'
    scores, indices = retrieve_chunks(query_1, embeddings_2, embedding_model_1)
    (scores, indices)
    return (query_1,)


@app.cell
def _(
    chunk_embedding_df_1,
    embedding_model_1,
    embeddings_2,
    print_topk,
    query_1,
):
    chunk_embedding_data = chunk_embedding_df_1.to_dict(orient='records')
    print_topk(query_1, embeddings_2, chunk_embedding_data, embedding_model_1)
    return (chunk_embedding_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Prepare LLM for local generation""")
    return


@app.cell
def _(os):
    if not "COLAB_GPU" in os.environ:
        from dotenv import load_dotenv
        load_dotenv()
    return


@app.cell
def _(torch):
    gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_mem_gb = round(gpu_mem_bytes / (2 ** 30))

    print(f'Available GPU Memory: {gpu_mem_gb} GB')
    return (gpu_mem_gb,)


@app.cell
def _(gpu_mem_gb):
    if gpu_mem_gb < 5.1:
        print(f"Your available GPU memory is {gpu_mem_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
    elif gpu_mem_gb < 8.1:
        print(f"GPU memory: {gpu_mem_gb} GB | Recommended model: Gemma 2B in 4-bit precision.")
        use_quantization_config = True
        model_id = "google/gemma-2b-it"
    elif gpu_mem_gb < 19.0:
        print(f"GPU memory: {gpu_mem_gb} GB | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
        use_quantization_config = False
        model_id = "google/gemma-2b-it"
    elif gpu_mem_gb > 19.0:
        print(f"GPU memory: {gpu_mem_gb} GB | Recommend model: Gemma 7B in 4-bit or float16 precision.")
        use_quantization_config = False
        model_id = "google/gemma-7b-it"

    print(f"use_quantization_config set to: {use_quantization_config}")
    print(f"model_id set to: {model_id}")
    return model_id, use_quantization_config


@app.cell
def _(device_1, model_id, torch, use_quantization_config):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import is_flash_attn_2_available
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    if is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] > 8:
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'sdpa'
    print(f'[INFO] Using attention implementation: {attn_implementation}')
    model_id_1 = model_id
    print(f'[INFO] Using model id: {model_id_1}')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id_1)
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id_1, torch_dtype=torch.float16, quantization_config=quantization_config if use_quantization_config else None, low_cpu_mem_usage=False, attn_implementation=attn_implementation)
    if not use_quantization_config:
        llm_model.to(device_1)
    return AutoTokenizer, llm_model, tokenizer


@app.cell
def _(llm_model):
    llm_model
    return


@app.cell
def _(torch):
    def get_model_nparam(model: torch.nn.Module):
        return sum([param.numel() for param in model.parameters()])


    def get_model_memsize(model: torch.nn.Module):
        """Get how much memory a model takes up"""

        mem_params = sum(
            [param.nelement() * param.element_size() for param in model.parameters()]
        )
        mem_buffers = sum(
            [buf.nelement() * buf.element_size() for buf in model.buffers()]
        )

        model_mem_bytes = mem_params + mem_buffers
        model_mem_mb = model_mem_bytes / (1024 ** 2)
        model_mem_gb = model_mem_bytes / (1024 ** 3)

        return {
            "model_mem_bytes" : model_mem_bytes,
            "model_mem_mb" : round(model_mem_mb, 2),
            "model_mem_gb" : round(model_mem_gb, 2),
        }
    return get_model_memsize, get_model_nparam


@app.cell
def _(get_model_nparam, llm_model):
    # get the number of parameters in our model
    get_model_nparam(llm_model)
    return


@app.cell
def _(get_model_memsize, llm_model):
    # get the memory requirement of our model
    get_model_memsize(llm_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Generating text with the LLM""")
    return


@app.cell
def _(tokenizer):
    input_text = (
        "What are the macronutrients,"
        " and what roles do they play in the human body?"
    )

    print(
        f"Query: {input_text}"
    )

    dialog_template = [
        {
            "role" : "user",
            "content" : input_text,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        conversation=dialog_template,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(
        f"\nPrompt (formatted):\n{prompt}"
    )
    return input_text, prompt


@app.cell
def _(device_1, llm_model, prompt, tokenizer):
    input_ids = tokenizer(prompt, return_tensors='pt').to(device_1)
    print(f'Model input (tokenized):\n{input_ids}')
    outputs = llm_model.generate(**input_ids, max_new_tokens=256)
    print(f'Model output (tokens):\n{outputs[0]}\n')
    return (outputs,)


@app.cell
def _(outputs, tokenizer):
    # Decode the output tokens to text
    outputs_decoded = tokenizer.decode(outputs[0])
    print(
        f"Model output (decoded):\n{outputs_decoded}\n"
    )
    return (outputs_decoded,)


@app.cell
def _(input_text, outputs_decoded, prompt):
    format_output = (
        lambda text: text
            .replace(prompt, '')
            .replace('<bos>', '')
            .replace('<eos>', '')
    )

    print(f"Input Text: {input_text}\n")
    print(f"Output Text:\n{format_output(outputs_decoded)}")
    return (format_output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Augmenting prompt with contextual chunks""")
    return


@app.cell
def _(AutoTokenizer, textwrap):
    def print_wrapped_1(text, wrap_length=79):
        """
        New print_wrapped version that respect the
        indentations of the LLM output and the prompt
        """
        for line in text.splitlines():
            indent = len(line) - len(line.lstrip())
            wrapped = textwrap.fill(line, width=wrap_length, subsequent_indent=' ' * indent, replace_whitespace=False, drop_whitespace=False)
            print(wrapped)

    def prompt_builder(query: str, context: list[dict], tokenizer: AutoTokenizer) -> str:
        """
        Augments query with text-based context.
        """
        context = '- ' + '\n- '.join([item['chunk'] for item in context])
        base_prompt = "Based on the following context items, please answer the query.\nGive yourself room to think by extracting relevant passages from the context before answering the query.\nDon't return the thinking, only return the answer.\nMake sure your answers are as explanatory as possible.\nUse the following examples as reference for the ideal answer style.\n\nExample 1:\nQuery: What are the fat-soluble vitamins?\nAnswer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.\n\nExample 2:\nQuery: What are the causes of type 2 diabetes?\nAnswer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.\n\nExample 3:\nQuery: What is the importance of hydration for physical performance?\nAnswer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.\n\nNow use the following context items to answer the user query:\n{context}\n\nRelevant passages: <extract relevant passages from the context here>\nUser query: {query}\nAnswer:"
        augmented = base_prompt.format(context=context, query=query)
        dialog_template = [{'role': 'user', 'content': augmented}]
        prompt = tokenizer.apply_chat_template(conversation=dialog_template, tokenize=False, add_generation_prompt=True)
        return prompt
    return print_wrapped_1, prompt_builder


@app.cell
def _():
    # Nutrition-style questions generated with GPT4
    gpt4_questions = [
        "What are the macronutrients, and what roles do they play in the human body?",
        "How do vitamins and minerals differ in their roles and importance for health?",
        "Describe the process of digestion and absorption of nutrients in the human body.",
        "What role does fibre play in digestion? Name five fibre containing foods.",
        "Explain the concept of energy balance and its importance in weight management."
    ]

    # Manually created question list
    manual_questions = [
        "How often should infants be breastfed?",
        "What are symptoms of pellagra?",
        "How does saliva help with digestion?",
        "What is the RDI for protein per day?",
        "water soluble vitamins"
    ]

    query_list = gpt4_questions + manual_questions
    return (query_list,)


@app.cell
def _(
    chunk_embedding_data,
    embedding_model_1,
    embeddings_2,
    print_wrapped_1,
    prompt_builder,
    query_list,
    random,
    retrieve_chunks,
    tokenizer,
):
    query_2 = random.choice(query_list)
    print(f'Query: {query_2}')
    scores_1, indices_1 = retrieve_chunks(query_2, embeddings_2, embedding_model_1)
    context = [chunk_embedding_data[i] for i in indices_1]
    prompt_1 = prompt_builder(query_2, context, tokenizer)
    print_wrapped_1(prompt_1)
    return prompt_1, query_2


@app.cell
def _(
    device_1,
    format_output,
    llm_model,
    print_wrapped_1,
    prompt_1,
    query_2,
    tokenizer,
):
    input_ids_1 = tokenizer(prompt_1, return_tensors='pt').to(device_1)
    outputs_1 = llm_model.generate(**input_ids_1, temperature=0.7, do_sample=True, max_new_tokens=512)
    output_text = tokenizer.decode(outputs_1[0])
    print(f'Query: {query_2}\n')
    print_wrapped_1(f'RAG answer:\n{format_output(output_text)}')
    return


if __name__ == "__main__":
    app.run()
