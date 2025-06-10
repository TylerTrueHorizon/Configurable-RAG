import os
import importlib.util

# Dynamically load the main module without shadowing the built-in `keyword`
SPEC = importlib.util.spec_from_file_location(
    "keyword_main", os.path.join(os.path.dirname(__file__), "..", "keyword", "main.py")
)
keyword_main = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(keyword_main)

load_documents = keyword_main.load_documents
build_corpus = keyword_main.build_corpus
search = keyword_main.search


def create_docs(tmpdir):
    files = {
        "doc1.txt": "apple orange banana",
        "doc2.txt": "carrot apple apple",
    }
    for name, text in files.items():
        with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as f:
            f.write(text)
    return files


def test_load_documents(tmp_path):
    create_docs(tmp_path)
    docs = load_documents(str(tmp_path))
    names = {name for name, _ in docs}
    assert names == {"doc1.txt", "doc2.txt"}


def test_query_returns_results(tmp_path):
    create_docs(tmp_path)
    docs = load_documents(str(tmp_path))
    corpus, meta = build_corpus(docs, chunk_size=2)
    results = search(corpus, meta, "apple", top_n=2)
    assert len(results) == 2
    assert all("apple" in chunk for _, _, _, chunk in results)
