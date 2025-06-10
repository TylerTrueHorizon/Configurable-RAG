import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from doc_parallel_rag import chunk_text, sub_chunks, gather_tasks

@pytest.fixture
def sample_text():
    return "word " * 500


def test_chunker(sample_text):
    chunks = chunk_text(sample_text, 'x', size=100, overlap=10)
    assert len(chunks) == 6
    assert chunks[0].meta['chunk_id'] == 0


def test_context_split():
    tokens = list(range(300))
    subs = sub_chunks(tokens, 120, overlap=20)
    assert len(subs) == 3
    assert all(len(s) <= 120 for s in subs)


def test_parallel_gather():
    async def work(x):
        await asyncio.sleep(0.01)
        return x

    results = asyncio.run(gather_tasks([work(i) for i in range(5)], max_workers=2))
    assert sorted(results) == list(range(5))
