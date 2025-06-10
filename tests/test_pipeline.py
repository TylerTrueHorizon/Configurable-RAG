import sys
import types
import numpy as np
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub tiktoken before importing the pipeline
class DummyEncoding:
    def encode(self, text):
        return text.split()
    def decode(self, tokens):
        return " ".join(tokens)

def encoding_for_model(name):
    return DummyEncoding()

tiktoken_stub = types.SimpleNamespace(encoding_for_model=encoding_for_model)
sys.modules['tiktoken'] = tiktoken_stub

# Stub faiss
class IndexFlatIP:
    def __init__(self, dim):
        self.vectors = np.zeros((0, dim), np.float32)
    def add(self, vecs):
        self.vectors = np.vstack([self.vectors, vecs])
    def search(self, q, k):
        sims = self.vectors @ q.T
        idx = np.argsort(sims[:,0])[::-1][:k]
        return sims[idx,0].reshape(1,-1), idx.reshape(1,-1)

def normalize_L2(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    x /= norms

faiss_stub = types.SimpleNamespace(IndexFlatIP=IndexFlatIP, normalize_L2=normalize_L2)
sys.modules['faiss'] = faiss_stub

import hyde_rag_pipeline


def fake_embed_texts(texts):
    data = [[len(t), sum(map(ord, t)) % 10, len(t.split())] for t in texts]
    return np.array(data, dtype=np.float32)


def test_load_document_txt(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("hello world")
    out = hyde_rag_pipeline.load_document(p)
    assert out == "hello world"


def test_hyde_query_flow(monkeypatch, tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("alpha beta gamma delta epsilon")

    text = hyde_rag_pipeline.load_document(p)
    chunks = hyde_rag_pipeline.chunk_text(text, tokens_per_chunk=2, overlap=1)
    all_chunks = []
    meta = []
    for i, (ck, cs, ce) in enumerate(chunks):
        all_chunks.append(ck)
        meta.append({
            "filename": str(p),
            "chunk_id": i,
            "char_start": cs,
            "char_end": ce,
            "text": ck,
        })

    monkeypatch.setattr(hyde_rag_pipeline, "embed_texts", fake_embed_texts)
    hyde_rag_pipeline.build_vector_store(all_chunks, meta)

    class DummyResp:
        def __init__(self, n):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=f"hypo {i}")) for i in range(n)]

    class DummyOpenAI:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda model, messages, n: DummyResp(n))
            )

    monkeypatch.setattr(hyde_rag_pipeline, "_OPENAI_AVAILABLE", True)
    monkeypatch.setattr(hyde_rag_pipeline, "OpenAI", DummyOpenAI)

    result = hyde_rag_pipeline.hyde_query("query", k_chunks=2, n_hypo=2)

    assert len(result["chunks"]) == 2
    assert len(result["hypo_docs"]) == 2
    assert result["scores"] == sorted(result["scores"], reverse=True)
