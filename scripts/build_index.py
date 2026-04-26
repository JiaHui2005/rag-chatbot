import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List

import yaml
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
MERGED_CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "merged_chunks.jsonl"
DEFAULT_PERSIST_DIR = ROOT_DIR / "db" / "chroma"


class SimpleHashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_config() -> Dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_documents(records: List[Dict]) -> List[Document]:
    documents: List[Document] = []
    for record in records:
        metadata = dict(record["metadata"])
        metadata["chunk_id"] = record["chunk_id"]
        metadata["doc_id"] = record["doc_id"]
        metadata["chunk_index"] = record["chunk_index"]
        documents.append(Document(page_content=record["text"], metadata=metadata))
    return documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chroma index from merged RAG chunks.")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help="Directory where the Chroma index will be stored.",
    )
    parser.add_argument(
        "--collection-name",
        default="land_law_rag",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "huggingface", "simple"],
        default="auto",
        help="Embedding backend. 'auto' tries HuggingFace first, then falls back to a local hash embedding.",
    )
    return parser.parse_args()


def build_embeddings(embedding_backend: str, model_name: str) -> Embeddings:
    if embedding_backend == "simple":
        print("Using offline simple hash embeddings.")
        return SimpleHashEmbeddings()

    if embedding_backend in {"auto", "huggingface"}:
        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"local_files_only": embedding_backend == "huggingface"},
            )
        except Exception as exc:
            if embedding_backend == "huggingface":
                raise
            print(
                "HuggingFace embedding model is unavailable in this environment. "
                f"Falling back to offline simple hash embeddings. Reason: {exc}"
            )
            return SimpleHashEmbeddings()

    raise ValueError(f"Unsupported embedding backend: {embedding_backend}")


def main() -> int:
    args = parse_args()
    if not MERGED_CHUNKS_PATH.exists():
        print("Missing merged chunks. Run scripts/build_chunks.py first.")
        return 1

    config = load_config()
    embedding_model = config.get("rag", {}).get("embedding_model", "all-MiniLM-L6-v2")
    records = load_jsonl(MERGED_CHUNKS_PATH)
    documents = build_documents(records)

    embeddings = build_embeddings(args.embedding_backend, embedding_model)
    args.persist_dir.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(args.persist_dir),
        collection_name=args.collection_name,
    )
    vector_store.persist()

    print(
        f"Indexed {len(documents)} chunks into "
        f"{args.persist_dir.relative_to(ROOT_DIR)} ({args.collection_name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
