from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

import faiss
import numpy as np
from docx import Document
from pdfminer.high_level import extract_text as pdf_extract

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


REFLECTION_TOKENS: Dict[str, str] = {
    "RET1": "<RET_1>",
    "RET2": "<RET_2>",
    "RET3": "<RET_3>",
    "RET4": "<RET_4>",
    "RET5": "<RET_5>",
    "ISREL": "<IS_REL>",
    "ISUSE": "<IS_USE>",
    "ISVER": "<IS_VER>",
}


class WhitespaceTokenizer:
    """Fallback whitespace tokenizer."""

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}

    @property
    def n_vocab(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, *_: str, **__: str) -> List[int]:
        ids = []
        for tok in text.split():
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)
            ids.append(self.token_to_id[tok])
        return ids

    def decode(self, ids: List[int]) -> str:
        inv = {v: k for k, v in self.token_to_id.items()}
        return " ".join(inv.get(i, "") for i in ids)

    def add_special_tokens(self, token_map: Dict[str, int]) -> int:
        for tok, idx in token_map.items():
            if tok not in self.token_to_id:
                self.token_to_id[tok] = idx
        return len(token_map)


def get_tokenizer(model: str = "cl100k_base"):
    """Return tokenizer extended with reflection tokens."""

    if tiktoken:
        try:
            enc = tiktoken.get_encoding(model)
            mapping = {t: enc.n_vocab + i for i, t in enumerate(REFLECTION_TOKENS.values())}
            enc.add_special_tokens(mapping)
            return enc
        except Exception:
            pass
    tok = WhitespaceTokenizer()
    mapping = {t: tok.n_vocab + i for i, t in enumerate(REFLECTION_TOKENS.values())}
    tok.add_special_tokens(mapping)
    return tok


def _read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return pdf_extract(path)
    if ext == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, tokenizer, size: int = 400, overlap: int = 50) -> List[str]:
    """Chunk text with overlap."""

    ids = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(ids):
        end = start + size
        chunk_ids = ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids))
        start += size - overlap
    return chunks


def ingest_paths(paths: List[str], tokenizer) -> List[str]:
    texts = []
    for p in paths:
        raw = _read_file(p)
        texts.extend(chunk_text(raw, tokenizer))
    return texts


def _embed_openai(texts: List[str]) -> Optional[np.ndarray]:
    try:
        import openai

        resp = openai.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
            batch_size=512,
        )
        return np.array([r.embedding for r in resp.data], dtype="float32")
    except Exception:
        return None


def _embed_sbert(texts: List[str]) -> Optional[np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, batch_size=64, convert_to_numpy=True)
    except Exception:
        return None


def embed_texts(texts: List[str]) -> np.ndarray:
    arr = _embed_openai(texts)
    if arr is None:
        arr = _embed_sbert(texts)
    if arr is None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer().fit(texts)
        arr = vec.transform(texts).toarray().astype("float32")
    arr = arr.astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    return arr / norms


@dataclass
class VectorStore:
    texts: List[str]
    embeddings: np.ndarray
    index: faiss.IndexFlatIP

    @classmethod
    def from_texts(cls, texts: List[str]) -> "VectorStore":
        emb = embed_texts(texts)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return cls(texts=texts, embeddings=emb, index=index)

    def search(self, query: str, k: int) -> List[str]:
        q_emb = embed_texts([query])
        _, idx = self.index.search(q_emb, k)
        return [self.texts[i] for i in idx[0] if i < len(self.texts)]


def generate_llm(prompt: str, stream: bool = False) -> Generator[str, None, str]:
    try:
        import openai

        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
        )
        if stream:
            for chunk in resp:
                text = chunk.choices[0].delta.get("content", "")
                if text:
                    yield text
            return ""
        else:
            return resp.choices[0].message.content
    except Exception:
        if stream:
            yield prompt
            return ""
        return prompt


def parse_critiques(text: str) -> Dict[str, Optional[int]]:
    scores: Dict[str, Optional[int]] = {}
    for name in ["ISREL", "ISUSE", "ISVER"]:
        token = REFLECTION_TOKENS[name]
        m = re.search(re.escape(token) + r"(\d)", text)
        scores[name] = int(m.group(1)) if m else None
    return scores


def selfrag_query(
    prompt: str,
    store: VectorStore,
    max_iterations: int = 4,
    beam_width: int = 4,
    stream: bool = False,
) -> Dict[str, object]:
    """Run Self-RAG inference loop."""

    current = prompt
    answer_parts: List[str] = []
    retrieved: List[str] = []
    for _ in range(max_iterations):
        if stream:
            buff = []
            for t in generate_llm(current, stream=True):
                buff.append(t)
            text = "".join(buff)
        else:
            text = generate_llm(current)
        for key in ["RET1", "RET2", "RET3", "RET4", "RET5"]:
            tok = REFLECTION_TOKENS[key]
            if tok in text:
                k = int(key[-1])
                docs = store.search(prompt, k)
                retrieved.extend(docs)
                text = text.replace(tok, "")
                current += "\n" + "\n".join(docs)
        answer_parts.append(text)
        if any(REFLECTION_TOKENS[k] in text for k in ["ISREL", "ISUSE", "ISVER"]):
            break
    final = "".join(answer_parts)
    crit = parse_critiques(final)
    if crit.get("ISVER") is not None and crit["ISVER"] < 3 and max_iterations > 1:
        return selfrag_query(prompt, store, max_iterations - 1, beam_width, stream)
    return {"answer": final.strip(), "passages": retrieved, "critiques": crit}


def finetune_generator() -> None:
    print("Placeholder: generate synthetic training data with GPT-4")


def finetune_critic() -> None:
    print("Placeholder: fine-tune critic model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-RAG pipeline")
    parser.add_argument("paths", nargs="*", help="Paths to documents")
    parser.add_argument("--query", type=str, help="User query")
    parser.add_argument("--stream", action="store_true", help="Stream tokens")
    parser.add_argument("--finetune-generator", action="store_true")
    parser.add_argument("--finetune-critic", action="store_true")
    args = parser.parse_args()

    if args.finetune_generator:
        finetune_generator()
    if args.finetune_critic:
        finetune_critic()
    if args.paths:
        tok = get_tokenizer()
        texts = ingest_paths(args.paths, tok)
        store = VectorStore.from_texts(texts)
    else:
        store = VectorStore.from_texts([])
    if args.query:
        res = selfrag_query(args.query, store, stream=args.stream)
        if isinstance(res, dict):
            print(res)


if __name__ == "__main__":
    main()
