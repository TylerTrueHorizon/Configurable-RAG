import asyncio
from pathlib import Path
import types

import pytest

import audio_rag_pipeline as arp


def test_chunk_text(monkeypatch):
    monkeypatch.setattr(arp, "tiktoken", None)
    text = "hello world " * 100
    chunks = arp.chunk_text(text, size=50, overlap=10)
    assert chunks


def test_tts_generate(tmp_path, monkeypatch):
    def fake_speech_create(model, voice, input):
        class R:
            content = b"data"
        return R()
    monkeypatch.setattr(arp.openai.audio, "speech", types.SimpleNamespace(create=fake_speech_create))
    path = arp.tts_generate("hi", "voice", tmp_path)
    assert Path(path).exists()


def test_build_and_query(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    doc = tmp_path / "docs/test.txt"
    doc.write_text("hola " * 200)
    monkeypatch.setattr(arp, "tiktoken", None)
    monkeypatch.setattr(arp, "INDEX_FILE", tmp_path / "i.faiss")
    monkeypatch.setattr(arp, "META_FILE", tmp_path / "m.pkl")
    monkeypatch.setattr(arp, "CACHE_DIR", tmp_path / "cache")
    arp.CACHE_DIR.mkdir()
    monkeypatch.setattr(arp, "tts_generate", lambda *a, **k: tmp_path / "a.mp3")
    monkeypatch.setattr(arp, "embed_texts", lambda texts: (len(texts),) and __import__("numpy").ones((len(texts), 3), dtype="float32"))
    arp.build_index([doc], "voice")
    async def fake_call(*a, **k):
        return "ans"
    async def fake_syn(sub, q, l):
        return "final"
    monkeypatch.setattr(arp, "call_audio_llm", fake_call)
    monkeypatch.setattr(arp, "gpt_synthesis", fake_syn)
    res = asyncio.run(arp.audio_rag_query("q", top_n=1, max_workers=1))
    assert res["answer"] == "final"
