import os
import random

from tqdm import tqdm
import requests
from bs4 import BeautifulSoup


import concurrent.futures

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")


def fetch_page(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        return response.text

    return ""


def extract_links_from_page(url: str, base_url="https://docs.ray.io") -> list[str]:
    html_content = fetch_page(url)
    if html_content != "":
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]  # type: ignore
        if href.startswith("/"):
            full_url = base_url + href
            if full_url not in links:
                links.append(full_url)
    return links


def load_and_process_url(url: str) -> list[Document]:
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()

        for doc in documents:
            doc.metadata.update({"source": url, "source_type": "ray_documentation"})

        return documents
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return []


def chunk_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents


def process_documents(
    start_url: str = "https://docs.ray.io/latest",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_urls: int = 50,
    min_length: int = 50,
    sample_size: int | None = None,
) -> list[Document]:
    """Full pipeline: fetch docs, convert to Langchain docs, chunk"""

    print(f"Extracting links from {start_url}")
    doc_links = extract_links_from_page(start_url)

    if max_urls and len(doc_links) > max_urls:
        doc_links = doc_links[:max_urls]

    print(f"Found {len(doc_links)} links, start extracting docs & processing...")
    all_docs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {
            executor.submit(load_and_process_url, url): url for url in doc_links
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_url),
            total=len(doc_links),
            desc="Loading documents",
        ):
            url = future_to_url[future]
            try:
                docs = future.result()
                all_docs.extend(docs)
            except Exception as e:
                print(f"{url} generaated an exception: {e}")

    print(f"Loaded {len(all_docs)} documents")

    # Chunking step
    print(
        f"Chunking documents with chunk_size={chunk_size}, overlap={chunk_overlap}..."
    )
    chunked_docs = chunk_documents(all_docs, chunk_size, chunk_overlap)

    print(f"Created {len(chunked_docs)} documents")

    # Filter by length
    filtered_docs = [
        doc for doc in chunked_docs if len(doc.page_content.split()) >= min_length
    ]

    print(f"Filtered to {len(filtered_docs)} chunks with at least {min_length} words")

    # Sample if needed
    if sample_size and sample_size < len(filtered_docs):
        docs_to_process = random.sample(filtered_docs, sample_size)
        print(f"Sampled {sample_size} chunks for processing")
    else:
        docs_to_process = filtered_docs

    return docs_to_process


if __name__ == "__main__":
    ...
