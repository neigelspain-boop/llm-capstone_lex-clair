"""Automated ingestion entry point (rubric: ingestion pipeline, 2 pts).

Run with:
    uv run python -m ingestion.ingest --source path/to/data

Fills the knowledge base end to end: load raw source -> chunk -> index
into the configured vector store (src.rag.retriever.get_vector_store).
"""
import argparse

from src.rag.retriever import get_vector_store


def load_documents(source: str) -> list[dict]:
    # TODO: load the chosen dataset (file, API, dump, etc.) from `source`
    raise NotImplementedError


def chunk_documents(documents: list[dict]) -> list[dict]:
    # TODO: split into retrieval-sized chunks; see project.md's chunking notes
    raise NotImplementedError


def run(source: str) -> None:
    documents = load_documents(source)
    chunks = chunk_documents(documents)
    store = get_vector_store()
    store.index(chunks)
    print(f"Indexed {len(chunks)} chunks from {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into the knowledge base")
    parser.add_argument("--source", required=True, help="Path or URI of the raw dataset")
    args = parser.parse_args()
    run(args.source)
