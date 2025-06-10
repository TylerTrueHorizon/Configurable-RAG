#!/usr/bin/env python3
"""Doc-Parallel RAG script."""
import argparse
import asyncio
import glob
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Dict, Iterable, List

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    from openai import OpenAI, RateLimitError  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = RateLimitError = None

try:
    from pdfminer.high_level import extract_text  # type: ignore
except Exception:  # pragma: no cover
    extract_text = None

try:
    from docx import Document  # type: ignore
except Exception:  # pragma: no cover
    Document = None

try:
    from langchain.llms import LlamaCpp
except Exception:  # pragma: no cover - optional
    LlamaCpp = None


@dataclass
class Chunk:
    text: str
    meta: Dict[str, Any]


def load_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf" and extract_text:
        return extract_text(path)
    if ext == ".docx" and Document:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, src_path: str, size: int = 400, overlap: int = 50) -> List[Chunk]:
    if tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        decode = enc.decode
    else:  # pragma: no cover - fallback
        ids = text.split()
        decode = lambda toks: " ".join(toks)
    out: List[Chunk] = []
    step = size - overlap
    for i in range(0, len(ids), step):
        sub = ids[i : i + size]
        start = len(decode(ids[:i]))
        end = len(decode(ids[: i + len(sub)]))
        out.append(Chunk(decode(sub), {
            "src_path": src_path,
            "chunk_id": len(out),
            "char_start": start,
            "char_end": end,
        }))
    return out


def embed_texts(texts: List[str], client):
    import numpy as np  # local import
    embs: List[List[float]] = []
    for i in range(0, len(texts), 512):
        resp = client.embeddings.create(input=texts[i : i + 512], model="text-embedding-3-small")
        embs.extend(e.embedding for e in resp.data)
    arr = np.array(embs, dtype="float32")
    if faiss:
        faiss.normalize_L2(arr)
    return arr


def build_index(chunks: List[Chunk], idx_path: str = "index.faiss", meta_path: str = "meta.pkl", client=None) -> None:
    if OpenAI and client is None:
        client = OpenAI()
    arr = embed_texts([c.text for c in chunks], client)
    if faiss:
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        faiss.write_index(index, idx_path)
    else:
        index = None
    with open(meta_path, "wb") as f:
        pickle.dump([c.meta for c in chunks], f)


def load_index(idx_path: str = "index.faiss", meta_path: str = "meta.pkl"):
    idx = faiss.read_index(idx_path) if faiss else None
    return idx, pickle.load(open(meta_path, "rb"))


async def gather_tasks(coros: Iterable[Awaitable], max_workers: int) -> List:
    sem = asyncio.Semaphore(max_workers)

    async def runner(c: Awaitable):
        async with sem:
            return await c

    return await asyncio.gather(*(runner(c) for c in coros))


def sub_chunks(tokens: List[int], size: int, overlap: int = 50) -> List[List[int]]:
    step = size - overlap
    return [tokens[i : i + size] for i in range(0, len(tokens), step)]


async def ask_llm(messages: List[Dict[str, str]], model: str = "gpt-4o", stream: bool = False, client=None) -> str:
    if client is None and OpenAI:
        client = OpenAI()
    if client is None:
        raise RuntimeError("No LLM backend available")
    for attempt in range(3):
        try:
            if stream:
                resp = client.chat.completions.create(model=model, messages=messages, stream=True)
                out = ""
                for ch in resp:
                    token = ch.choices[0].delta.content or ""
                    out += token
                    print(token, end="", flush=True)
                print()
                return out
            resp = client.chat.completions.create(model=model, messages=messages)
            return resp.choices[0].message.content
        except RateLimitError:
            await asyncio.sleep(2**attempt)
    raise RuntimeError("LLM call failed")


async def doc_parallel_query(
    question: str,
    top_n: int = 8,
    ctx_frac: float = 0.40,
    max_workers: int = 16,
    model: str = "gpt-4o",
    ctx: int | None = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """Embed question, retrieve docs, ask in parallel and synthesize."""

    if not OpenAI:
        raise RuntimeError("OpenAI package required")
    client = OpenAI()
    index, meta = load_index()
    q_emb = embed_texts([question], client)[0]
    import numpy as np  # local import
    I = index.search(np.array([q_emb]), top_n)[1] if index else [[]]
    hits = [meta[i] for i in I[0]] if I else []
    docs: Dict[str, str] = {}
    for h in hits:
        if h["src_path"] not in docs:
            docs[h["src_path"]] = load_file(h["src_path"])
    model_ctx = ctx or client.models.retrieve(model).context_window
    enc = tiktoken.get_encoding("cl100k_base") if tiktoken else None

    async def worker(text: str) -> str:
        prompt = [{"role": "user", "content": text + "\n" + question}]
        return await ask_llm(prompt, model=model, client=client)

    coros = []
    for text in docs.values():
        ids = enc.encode(text) if enc else text.split()
        max_len = int(ctx_frac * model_ctx)
        if len(ids) <= max_len:
            coros.append(worker(text))
        else:
            for sub in sub_chunks(ids, max_len):
                part = enc.decode(sub) if enc else " ".join(sub)
                coros.append(worker(part))

    start = time.time()
    doc_answers = await gather_tasks(coros, max_workers)
    synth_prompt = [
        {"role": "system", "content": "Combine the following answers into one coherent response."},
        {"role": "user", "content": "\n\n".join(doc_answers)},
    ]
    final = await ask_llm(synth_prompt, model=model, stream=stream, client=client)
    return {"final_answer": final, "doc_answers": doc_answers, "timing_stats": {"seconds": time.time() - start}}


def load_and_chunk(paths: List[str]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for p in paths:
        text = load_file(p)
        chunks.extend(chunk_text(text, p))
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", help="Document paths or globs")
    ap.add_argument("--ask", help="Question")
    ap.add_argument("--top_n", type=int, default=8)
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()
    if args.docs:
        files = [f for pat in args.docs for f in glob.glob(pat)]
        build_index(load_and_chunk(files))
    if args.ask:
        res = asyncio.run(
            doc_parallel_query(
                args.ask,
                top_n=args.top_n,
                max_workers=args.max_workers,
                ctx=args.ctx,
                stream=args.stream,
            )
        )
        if not args.stream:
            print(res["final_answer"])


if __name__ == "__main__":
    main()
