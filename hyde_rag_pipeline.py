import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import tiktoken
import faiss

# Optional imports handled dynamically
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover - openai may not be installed
    OpenAI = None
    _OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:  # pragma: no cover - sentence-transformers may not be installed
    SentenceTransformer = None
    _ST_AVAILABLE = False

# Global holders for FAISS index and metadata
_INDEX = None
_METADATA = []


def load_document(path: Path) -> str:
    """Load text from supported document formats."""
    ext = path.suffix.lower()
    try:
        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8")
        if ext == ".pdf":
            from pdfminer.high_level import extract_text
            return extract_text(str(path))
        if ext == ".docx":
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:  # pragma: no cover - runtime failure
        print(f"Error parsing {path}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Unsupported file type: {path}", file=sys.stderr)
    sys.exit(1)


def chunk_text(text: str, tokens_per_chunk: int = 400, overlap: int = 50) -> list[tuple[str, int, int]]:
    """Split text into token chunks returning (chunk, char_start, char_end)."""
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = enc.encode(text)
    chunks = []
    start_tok = 0
    while start_tok < len(tokens):
        end_tok = min(start_tok + tokens_per_chunk, len(tokens))
        chunk_tokens = tokens[start_tok:end_tok]
        chunk_str = enc.decode(chunk_tokens)
        char_start = len(enc.decode(tokens[:start_tok]))
        char_end = char_start + len(chunk_str)
        chunks.append((chunk_str, char_start, char_end))
        start_tok += tokens_per_chunk - overlap
    return chunks


def _embed_openai(texts: list[str], model: str = "text-embedding-3-small", batch_size: int = 512) -> list[np.ndarray]:
    client = OpenAI()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend(np.array(d.embedding, dtype=np.float32) for d in resp.data)
    return out


def _embed_fallback(texts: list[str], model: str = "all-MiniLM-L6-v2") -> list[np.ndarray]:
    if not _ST_AVAILABLE:
        raise RuntimeError("sentence-transformers is required for fallback embeddings")
    st = SentenceTransformer(model)
    embs = st.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
    return [emb.astype(np.float32) for emb in embs]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts using OpenAI, falling back to SentenceTransformer."""
    if _OPENAI_AVAILABLE:
        try:
            vecs = _embed_openai(texts)
            return np.stack(vecs)
        except Exception as exc:  # pragma: no cover - runtime failure
            print(f"OpenAI embedding failed: {exc}", file=sys.stderr)
    if _ST_AVAILABLE:
        vecs = _embed_fallback(texts)
        return np.stack(vecs)
    print("No embedding backend available", file=sys.stderr)
    sys.exit(1)


def build_vector_store(chunks: list[str], metadata: list[dict]) -> faiss.IndexFlatIP:
    """Create FAISS index from text chunks with metadata."""
    embeddings = embed_texts(chunks)
    faiss.normalize_L2(embeddings)
    if embeddings.nbytes > 1_073_741_824:
        warnings.warn("Vector store may exceed 1 GB RAM usage")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    global _INDEX, _METADATA
    _INDEX = index
    _METADATA = metadata
    return index


def hyde_query(question: str, k_chunks: int = 5, n_hypo: int = 3, gen_model: str = "gpt-4o") -> dict:
    """Perform HyDE retrieval over the indexed chunks."""
    if _INDEX is None:
        raise RuntimeError("Vector store not built")
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI client required for HyDE generation")

    client = OpenAI()
    resp = client.chat.completions.create(
        model=gen_model,
        messages=[{"role": "user", "content": question}],
        n=n_hypo,
    )
    hypos = [choice.message.content.strip() for choice in resp.choices]
    hypo_embs = embed_texts(hypos)
    faiss.normalize_L2(hypo_embs)
    query_vec = hypo_embs.mean(axis=0, keepdims=True)
    faiss.normalize_L2(query_vec)

    scores, indices = _INDEX.search(query_vec.astype(np.float32), k_chunks)
    results = []
    for idx in indices[0]:
        meta = _METADATA[idx]
        results.append(meta)

    return {"scores": scores[0].tolist(), "chunks": results, "hypo_docs": hypos}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="HyDE RAG pipeline")
    parser.add_argument("files", nargs="+", help="Input documents (.txt, .md, .pdf, .docx)")
    parser.add_argument("--query", help="Optional query to run with HyDE")
    parser.add_argument("--k", type=int, default=5, help="Chunks to retrieve")
    args = parser.parse_args(argv)

    all_chunks = []
    metadata = []
    for file_path in args.files:
        path = Path(file_path)
        text = load_document(path)
        cks = chunk_text(text)
        for i, (chunk, start, end) in enumerate(cks):
            all_chunks.append(chunk)
            metadata.append({
                "filename": str(path),
                "chunk_id": i,
                "char_start": start,
                "char_end": end,
                "text": chunk,
            })

    build_vector_store(all_chunks, metadata)

    if args.query:
        result = hyde_query(args.query, k_chunks=args.k)
        for i, meta in enumerate(result["chunks"]):
            print(f"# Chunk {i} score={result['scores'][i]:.3f}")
            print(f"From {meta['filename']} [{meta['char_start']}-{meta['char_end']}]:")
            print(meta['text'][:200], "...\n")


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main(sys.argv[1:])
