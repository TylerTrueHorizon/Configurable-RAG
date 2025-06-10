import argparse
import os
from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi


def load_documents(doc_dir: str) -> List[Tuple[str, str]]:
    """Load text from all files in `doc_dir`."""
    docs = []
    for name in os.listdir(doc_dir):
        path = os.path.join(doc_dir, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((name, f.read()))
    return docs


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """Split `text` into chunks of approximately `chunk_size` words."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def build_corpus(docs: List[Tuple[str, str]], chunk_size: int):
    corpus = []
    metadata = []
    for filename, text in docs:
        chunks = chunk_text(text, chunk_size)
        for idx, chunk in enumerate(chunks):
            corpus.append(chunk)
            metadata.append({"file": filename, "chunk_index": idx})
    return corpus, metadata


def search(corpus: List[str], metadata: List[dict], query: str, top_n: int):
    """Rank corpus chunks using Okapi BM25."""
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    scores = np.maximum(scores, 0)
    best_indices = np.argsort(scores)[::-1][:top_n]
    results = [
        (scores[i], metadata[i]["file"], metadata[i]["chunk_index"], corpus[i])
        for i in best_indices
    ]
    return results


def main():
    parser = argparse.ArgumentParser(description="Keyword search in documents")
    parser.add_argument("--docs", required=True, help="Folder with documents")
    parser.add_argument("--query", required=True, help="Keyword query")
    parser.add_argument("--top_n", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of words per chunk",
    )
    args = parser.parse_args()

    docs = load_documents(args.docs)
    corpus, meta = build_corpus(docs, args.chunk_size)
    results = search(corpus, meta, args.query, args.top_n)

    if not results:
        print("No matches found.")
        return

    for score, filename, idx, text in results:
        print(f"\nFile: {filename} | Chunk: {idx} | Score: {score:.4f}")
        print(text)


if __name__ == "__main__":
    main()
