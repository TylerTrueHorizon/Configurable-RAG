import argparse
import glob
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()


def _compute_hash(paths: List[str]) -> str:
    """Return a hash representing the set of input files."""
    m = hashlib.md5()
    for p in sorted(paths):
        st = os.stat(p)
        m.update(p.encode())
        m.update(str(st.st_mtime_ns).encode())
    return m.hexdigest()


def _ingest_documents(paths: List[str]):
    """Load files and split into nodes."""
    from llama_index.readers.file import SimpleDirectoryReader
    from llama_index.core.node_parser import TokenTextSplitter

    reader = SimpleDirectoryReader(input_files=paths)
    docs = reader.load_data()
    splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.get_nodes_from_documents(docs)


def _build_indices(paths: List[str], storage_dir: Path):
    """Load or build indices and return them with nodes."""
    from llama_index.core import VectorStoreIndex, SummaryIndex
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.settings import Settings

    storage_dir.mkdir(exist_ok=True)
    hash_path = storage_dir / "docs.hash"
    vector_dir = storage_dir / "vector"
    summary_dir = storage_dir / "summary"

    doc_hash = _compute_hash(paths)
    rebuild = True
    if hash_path.exists() and hash_path.read_text() == doc_hash:
        if (vector_dir / "index.json").exists() and (summary_dir / "index.json").exists():
            rebuild = False

    nodes = _ingest_documents(paths)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    if rebuild:
        vector_sc = StorageContext.from_defaults(persist_dir=str(vector_dir))
        vector_index = VectorStoreIndex(nodes, storage_context=vector_sc)
        vector_sc.persist()
        summary_sc = StorageContext.from_defaults(persist_dir=str(summary_dir))
        summary_index = SummaryIndex(nodes, storage_context=summary_sc)
        summary_sc.persist()
        hash_path.write_text(doc_hash)
    else:
        vector_sc = StorageContext.from_defaults(persist_dir=str(vector_dir))
        summary_sc = StorageContext.from_defaults(persist_dir=str(summary_dir))
        vector_index = load_index_from_storage(vector_sc)
        summary_index = load_index_from_storage(summary_sc)

    return nodes, vector_index, summary_index


def _setup_composite_retriever(vector_index, bm25_retr, summary_index):
    """Create or connect to a composite retriever on LlamaCloud."""
    from llama_cloud import CompositeRetrievalMode
    from llama_index.indices.managed.llama_cloud import LlamaCloudCompositeRetriever

    project = os.environ.get("PROJECT", "default")
    composite = LlamaCloudCompositeRetriever(
        name="Knowledge Agent",
        project_name=project,
        create_if_not_exists=True,
        mode=CompositeRetrievalMode.ROUTED,
        rerank_top_n=8,
    )

    vec_retr = vector_index.as_retriever(retrieval_mode="auto_routed")
    bm25_index = bm25_retr
    bm25_retriever = bm25_index.as_retriever(retrieval_mode="auto_routed")
    summ_retr = summary_index.as_retriever(retrieval_mode="auto_routed")

    composite.add_index(vec_retr.index, description="Dense semantic search over technical PDFs")
    composite.add_index(bm25_retriever.index, description="Keyword/BM25 search for exact matches")
    composite.add_index(summ_retr.index, description="High-level summaries for abstract queries")
    return composite

def composite_query(question: str, top_k: int = 6, stream: bool = False) -> Dict[str, Any]:



    response_synth = get_response_synthesizer(streaming=stream)
    engine = RetrieverQueryEngine(retriever=composite_retriever, response_synthesizer=response_synth)
    response = engine.query(question)

    nodes = []
    for node in response.source_nodes[:top_k]:
        nodes.append({
            "text": node.node.get_content(),
            "retrieval_mode": node.node.metadata.get("retrieval_mode", ""),
        })
    return {"answer": str(response), "supporting_nodes": nodes, "retrieval_trail": response.metadata}


def _pretty_print(nodes: List[Dict[str, str]]) -> None:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Mode", style="cyan", width=12)
    table.add_column("Text", style="white")
    for n in nodes:
        table.add_row(n["retrieval_mode"], n["text"].replace("\n", " ")[:200])
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Composite Retrieval Demo")
    parser.add_argument("--docs", nargs="+", required=True, help="Document file globs")
    parser.add_argument("--ask", required=True, help="Question to ask")
    parser.add_argument("--stream", action="store_true", help="Stream the answer")
    args = parser.parse_args()

    file_paths: List[str] = []
    for pattern in args.docs:
        file_paths.extend(glob.glob(pattern))

    nodes, vec_index, sum_index = _build_indices(file_paths, Path("storage"))
    from llama_index.legacy.retrievers import BM25Retriever

    bm25_retr = BM25Retriever.from_defaults(nodes=nodes)
    composite_retriever = _setup_composite_retriever(vec_index, bm25_retr, sum_index)
    result = composite_query(args.ask, stream=args.stream)
    console.print(f"\n[bold]Answer:[/bold] {result['answer']}\n")
    _pretty_print(result["supporting_nodes"])
