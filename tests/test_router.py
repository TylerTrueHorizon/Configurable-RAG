import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import types
import composite_retrieval_pipeline as crp

class DummyRetriever:
    def retrieve(self, query):
        mode = "bm25" if "." in query else "vector"
        node = types.SimpleNamespace(get_content=lambda: "dummy", metadata={"retrieval_mode": mode})
        return [types.SimpleNamespace(node=node, score=1.0)]

class DummyEngine:
    def __init__(self, retriever, response_synthesizer=None):
        self.retriever = retriever
    def query(self, question):
        return types.SimpleNamespace(
            source_nodes=self.retriever.retrieve(question),
            metadata={},
            __str__=lambda self: "answer"
        )

def test_routing(monkeypatch):
    monkeypatch.setattr(crp, "RetrieverQueryEngine", DummyEngine)
    monkeypatch.setattr(crp, "get_response_synthesizer", lambda streaming=False: None)
    crp.composite_retriever = DummyRetriever()

    res1 = crp.composite_query("file.pdf")
    assert res1["supporting_nodes"][0]["retrieval_mode"] == "bm25"
    res2 = crp.composite_query("tell me about ai")
    assert res2["supporting_nodes"][0]["retrieval_mode"] == "vector"
