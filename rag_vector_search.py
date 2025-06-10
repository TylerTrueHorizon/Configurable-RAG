#!/usr/bin/env python3
"""Self-contained RAG vector search."""

from __future__ import annotations

import os
import sys
import json
from typing import List, Dict

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    print("faiss not installed", file=sys.stderr)
    raise

try:
    import tiktoken
    ENC = tiktoken.encoding_for_model("text-embedding-3-small")
except Exception:  # pragma: no cover - allow offline usage
    class _DummyEnc:
        def encode(self, text: str):
            return text.split()

        def decode(self, tokens):
            return " ".join(tokens)

    ENC = _DummyEnc()
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

try:
    import openai
except Exception:
    openai = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

# Global resources
INDEX: faiss.IndexFlatIP | None = None
META: List[Dict] = []
ST_MODEL: SentenceTransformer | None = None


def load_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in {".txt", ".md"}:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        if ext == ".pdf":
            return pdf_extract_text(path)
        if ext == ".docx":
            return "\n".join(p.text for p in Document(path).paragraphs)
    except Exception as e:  # pragma: no cover - handle below
        raise RuntimeError(f"Failed to read {path}: {e}")
    raise RuntimeError(f"Unsupported file type: {path}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 20) -> List[str]:
    tokens = ENC.encode(text)
    chunks: List[str] = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(ENC.decode(chunk_tokens))
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    global ST_MODEL
    if os.getenv("FAKE_EMBEDDINGS"):
        arr = np.array([[hash(t) % 1000] for t in texts], dtype=np.float32)
    elif os.getenv("OPENAI_API_KEY") and openai is not None:
        res = openai.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
            encoding_format="float",
        )
        arr = np.array([r.embedding for r in res.data], dtype=np.float32)
    else:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available")
        if ST_MODEL is None:
            ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        arr = ST_MODEL.encode(texts, batch_size=512, show_progress_bar=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.clip(norms, 1e-10, None)
    return arr.astype("float32")


def build_index(files: List[str]) -> None:
    global INDEX, META
    all_chunks: List[str] = []
    meta: List[Dict] = []
    for fname in files:
        try:
            text = load_file(fname)
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        chunks = chunk_text(text)
        for cid, chunk in enumerate(chunks):
            char_start = text.find(chunk)
            char_end = char_start + len(chunk)
            meta.append(
                {
                    "filename": os.path.basename(fname),
                    "chunk_id": cid,
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )
        all_chunks.extend(chunks)
    embeds = embed_texts(all_chunks)
    INDEX = faiss.IndexFlatIP(embeds.shape[1])
    INDEX.add(embeds)
    META = meta
    size_bytes = embeds.nbytes
    if size_bytes > 1_000_000_000:
        print(
            f"Index uses {size_bytes / 1_000_000:.2f} MB RAM. Consider using an IVF-PQ index.",
            file=sys.stderr,
        )


def query(text: str, k: int = 5) -> List[Dict]:
    if INDEX is None:
        raise RuntimeError("Index not built")
    vec = embed_texts([text])
    distances, indices = INDEX.search(vec, k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        item = META[idx].copy()
        item["score"] = float(score)
        results.append(item)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python rag_vector_search.py <files...>", file=sys.stderr)
        sys.exit(1)
    build_index(sys.argv[1:])
    print("Ready. Enter queries (Ctrl-D to exit).")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        hits = query(line)
        print(json.dumps(hits, indent=2))


if __name__ == "__main__":
    main()
