import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import doc_parallel_rag as dpr

class DummyIndex:
    def add(self, arr):
        pass

    def search(self, arr, top_n):
        return None, [[0]]

dummy_faiss = type(
    "faiss",
    (),
    {
        "IndexFlatIP": lambda dim: DummyIndex(),
        "write_index": lambda index, path: None,
        "read_index": lambda path: DummyIndex(),
        "normalize_L2": lambda arr: None,
    },
)

async def fake_ask_llm(*args, **kwargs):
    return "final"


class DummyArray(list):
    def __init__(self, data):
        super().__init__(data)
        self._shape = (len(data), len(data[0]) if data else 0)

    @property
    def shape(self):
        return self._shape


def dummy_embed(texts, client=None):
    return DummyArray([[1.0, 1.0, 1.0] for _ in texts])


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    doc = tmp_path / "t.txt"
    doc.write_text("hello world")

    monkeypatch.setattr(dpr, "faiss", dummy_faiss)
    monkeypatch.setattr(dpr, "embed_texts", dummy_embed)
    monkeypatch.setattr(dpr, "ask_llm", fake_ask_llm)
    DummyClient = type(
        "DummyClient",
        (),
        {"models": type("M", (), {"retrieve": staticmethod(lambda m: type("C", (), {"context_window": 100})())})},
    )
    monkeypatch.setattr(dpr, "OpenAI", DummyClient)
    sys.modules["numpy"] = type("np", (), {"array": lambda x, dtype=None: x})

    chunks = dpr.load_and_chunk([str(doc)])
    dpr.build_index(chunks)

    res = asyncio.run(dpr.doc_parallel_query("hi", top_n=1))
    assert res["final_answer"] == "final"
    assert res["doc_answers"] == ["final"]
