#!/usr/bin/env python3
"""Audio-centric RAG pipeline."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

try:
    from pdfminer.high_level import extract_text as pdf_extract
except Exception:  # pragma: no cover
    pdf_extract = None

try:
    from docx import Document as DocxDocument
except Exception:  # pragma: no cover
    DocxDocument = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
INDEX_FILE = Path("index.faiss")
META_FILE = Path("metadata.pkl")


def sha256(data: str) -> str:
    """Return hex digest of SHA-256."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def parse_document(path: Path) -> str:
    """Parse supported document types."""
    try:
        if path.suffix.lower() in {".txt", ".md"}:
            return path.read_text()
        if path.suffix.lower() == ".pdf" and pdf_extract:
            return pdf_extract(str(path))
        if path.suffix.lower() == ".docx" and DocxDocument:
            return "\n".join(p.text for p in DocxDocument(str(path)).paragraphs)
    except Exception:
        pass
    raise ValueError(f"Failed to parse {path}")


def chunk_text(text: str, size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping token chunks."""
    if not tiktoken:
        return [text]
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = enc.decode(tokens[i : i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def retry_async(func):
    async def wrapper(*args, **kwargs):
        delay = 1
        for _ in range(3):
            try:
                return await func(*args, **kwargs)
            except Exception:
                await asyncio.sleep(delay)
                delay *= 2
        raise

    return wrapper


def tts_generate(text: str, voice: str, out_dir: Path) -> str:
    """Generate or load cached TTS audio for text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key = sha256(text)
    path = out_dir / f"{key}.mp3"
    if path.exists():
        return str(path)
    if not openai:
        path.write_bytes(b"")
        return str(path)
    for attempt in range(3):
        try:
            response = openai.audio.speech.create(model="tts-1", voice=voice, input=text)
            with open(path, "wb") as f:
                f.write(response.content)
            break
        except Exception:
            if attempt == 2:
                raise
    return str(path)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts using OpenAI or a fallback model."""
    if openai:
        try:
            res = openai.embeddings.create(model="text-embedding-3-small", input=texts)
            arr = np.array([e["embedding"] for e in res.data], dtype="float32")
        except Exception:
            arr = None
    else:
        arr = None
    if arr is None and SentenceTransformer:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        arr = model.encode(texts, batch_size=32, convert_to_numpy=True)
    if arr is None:
        arr = np.random.randn(len(texts), 384).astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-12)
    return arr.astype("float32")


def build_index(files: List[Path], voice: str) -> None:
    """Process documents and build vector index."""
    metadatas = []
    embeddings = []
    offset = 0
    for path in files:
        try:
            text = parse_document(path)
        except Exception:
            print(f"Could not parse {path}")
            continue
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            audio_path = tts_generate(chunk, voice, Path("audio") / path.stem)
            emb = embed_texts([chunk])[0]
            embeddings.append(emb)
            metadatas.append({
                "chunk_id": offset + idx,
                "audio_path": audio_path,
                "doc_name": path.name,
            })
        offset += len(chunks)
    if not embeddings:
        return
    d = len(embeddings[0])
    index = faiss.IndexFlatIP(d)
    index.add(np.stack(embeddings))
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_bytes(pickle.dumps(metadatas))


def load_index() -> tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load FAISS index and metadata."""
    index = faiss.read_index(str(INDEX_FILE))
    metas = pickle.loads(META_FILE.read_bytes())
    return index, metas


async def call_audio_llm(path: str, question: str, sem: asyncio.Semaphore, model: str) -> str:
    """Call audio LLM with retries and caching."""
    key = sha256(path + question)
    cache = CACHE_DIR / f"{key}.txt"
    if cache.exists():
        return cache.read_text()

    @retry_async
    async def call() -> str:
        if not openai:
            return ""
        with open(path, "rb") as f:
            params = {"audio": f, "query": question, "model": model}
            res = await openai.AsyncAudio.llm.create(**params)
        return res["text"]

    async with sem:
        text = await call()
    cache.write_text(text)
    return text


@retry_async
def gpt_synthesis(sub_answers: List[str], question: str, language: str | None) -> str:
    """Synthesize final answer."""
    if not openai:
        return ""
    prompt = (
        "Synthesize the following answers into one coherent response."\
        "\nQuestion: " + question + "\n" + "\n".join(sub_answers)
    )
    res = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )
    answer = res.choices[0].message.content
    if language and language != "en":
        trans = openai.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"Translate to {language}: {answer}"}]
        )
        answer = trans.choices[0].message.content
    return answer


async def audio_rag_query(question: str, top_n: int = 8, max_workers: int = 12, audio_llm: str = "AudioPaLM", language: str | None = None) -> Dict[str, Any]:
    """Run retrieval and audio-LLM pipeline."""
    index, metas = load_index()
    q_emb = embed_texts([question])
    _, idxs = index.search(q_emb, top_n)
    paths = []
    consulted = []
    for i in idxs[0]:
        meta = metas[i]
        path = meta["audio_path"]
        if path not in paths:
            paths.append(path)
        consulted.append(meta["doc_name"])
    sem = asyncio.Semaphore(max_workers)
    tasks = [call_audio_llm(p, question, sem, audio_llm) for p in paths[:max_workers]]
    sub_answers = await asyncio.gather(*tasks)
    answer = await gpt_synthesis(sub_answers, question, language)
    return {"answer": answer, "sub_answers": sub_answers, "consulted_files": consulted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", nargs="*")
    parser.add_argument("--voice", default="tts-1-voice")
    parser.add_argument("--ask")
    parser.add_argument("--top_n", type=int, default=8)
    parser.add_argument("--language")
    args = parser.parse_args()
    if args.docs:
        files = [Path(p) for pat in args.docs for p in sorted(Path().glob(pat))]
        build_index(files, args.voice)
    if args.ask:
        res = asyncio.run(audio_rag_query(args.ask, args.top_n, language=args.language))
        print(res["answer"])


if __name__ == "__main__":  # pragma: no cover
    main()
