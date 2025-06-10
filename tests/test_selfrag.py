import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import selfrag_pipeline as sp


def fake_embed(texts):
    return np.stack([np.array([len(t), len(t) + 1], dtype="float32") for t in texts])


def test_tokenizer_extension():
    tok = sp.get_tokenizer()
    for val in sp.REFLECTION_TOKENS.values():
        if isinstance(tok, sp.WhitespaceTokenizer):
            assert val in tok.token_to_id
        else:
            assert val in tok._special_tokens or val in getattr(tok, 'special_tokens_set', set())


def test_retrieval_on_demand(monkeypatch):
    monkeypatch.setattr(sp, "embed_texts", fake_embed)
    docs = ["doc one", "doc two"]
    store = sp.VectorStore.from_texts(docs)

    def fake_generate(prompt: str, stream: bool = False):
        return "Here is an answer " + sp.REFLECTION_TOKENS["RET1"] + " done"

    monkeypatch.setattr(sp, "generate_llm", fake_generate)
    res = sp.selfrag_query("question", store)
    assert res["passages"]


def test_critique_parsing(monkeypatch):
    monkeypatch.setattr(sp, "embed_texts", fake_embed)
    store = sp.VectorStore.from_texts(["a"])

    def fake_generate(prompt: str, stream: bool = False):
        return (
            "final "
            + sp.REFLECTION_TOKENS["ISREL"]
            + "5 "
            + sp.REFLECTION_TOKENS["ISUSE"]
            + "4 "
            + sp.REFLECTION_TOKENS["ISVER"]
            + "2"
        )

    monkeypatch.setattr(sp, "generate_llm", fake_generate)
    res = sp.selfrag_query("question", store)
    assert res["critiques"]["ISVER"] == 2
