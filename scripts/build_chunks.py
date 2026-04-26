import json
import re
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
LEGAL_INPUT_PATH = ROOT_DIR / "data" / "cleaned" / "legal_docs.jsonl"
FAQ_INPUT_PATH = ROOT_DIR / "data" / "cleaned" / "faq_web.jsonl"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
LEGAL_OUTPUT_PATH = PROCESSED_DIR / "legal_chunks.jsonl"
FAQ_OUTPUT_PATH = PROCESSED_DIR / "faq_chunks.jsonl"
MERGED_OUTPUT_PATH = PROCESSED_DIR / "merged_chunks.jsonl"

MAX_WORDS_LEGAL = 220
MAX_WORDS_FAQ = 320


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_by_word_budget(text: str, max_words: int) -> List[str]:
    paragraphs = [segment.strip() for segment in text.split("\n") if segment.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if current and current_words + len(words) > max_words:
            chunks.append("\n".join(current).strip())
            current = []
            current_words = 0
        current.append(paragraph)
        current_words += len(words)

    if current:
        chunks.append("\n".join(current).strip())

    return chunks or [text.strip()]


def legal_prefix(record: Dict) -> str:
    metadata = record["metadata"]
    parts = [record["title"]]
    if metadata.get("chapter"):
        parts.append(metadata["chapter"])
    if metadata.get("article"):
        parts.append(metadata["article"])
    if metadata.get("chapter_title"):
        parts.append(metadata["chapter_title"])
    return ". ".join(parts)


def faq_prefix(record: Dict) -> str:
    return f"Câu hỏi: {record['metadata']['question']}"


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_chunk_record(record: Dict, chunk_text: str, chunk_index: int) -> Dict:
    metadata = record["metadata"]
    return {
        "chunk_id": f"{record['doc_id']}_chunk_{chunk_index}",
        "doc_id": record["doc_id"],
        "chunk_index": chunk_index,
        "text": chunk_text,
        "metadata": {
            "source_type": record["source_type"],
            "title": record["title"],
            "domain": metadata["domain"],
            "topic": metadata.get("topic"),
            "article": metadata.get("article"),
            "clause": metadata.get("clause"),
            "question": metadata.get("question"),
            "source_url": metadata.get("source_url"),
            "source_path": metadata.get("source_path"),
            "authority_level": metadata.get("authority_level"),
            "tags": metadata.get("tags", []),
        },
    }


def chunk_legal_records(records: List[Dict]) -> List[Dict]:
    chunks: List[Dict] = []
    for record in records:
        prefix = legal_prefix(record)
        body = normalize_text(record["content"])
        subchunks = split_by_word_budget(body, MAX_WORDS_LEGAL)
        for index, subchunk in enumerate(subchunks):
            text = f"{prefix}. {subchunk}".strip()
            chunks.append(build_chunk_record(record, text, index))
    return chunks


def chunk_faq_records(records: List[Dict]) -> List[Dict]:
    chunks: List[Dict] = []
    for record in records:
        prefix = faq_prefix(record)
        body = normalize_text(record["metadata"]["answer"])
        subchunks = split_by_word_budget(body, MAX_WORDS_FAQ)
        for index, subchunk in enumerate(subchunks):
            if index == 0:
                text = f"{prefix}\nTrả lời: {subchunk}"
            else:
                text = f"{prefix}\nTrả lời (tiếp): {subchunk}"
            chunks.append(build_chunk_record(record, text, index))
    return chunks


def main() -> int:
    if not LEGAL_INPUT_PATH.exists():
        print("Missing cleaned legal docs. Run scripts/clean_legal_docs.py first.")
        return 1
    if not FAQ_INPUT_PATH.exists():
        print(
            "Missing cleaned FAQ docs. Run scripts/crawl_faq_web.py and "
            "then scripts/clean_faq_web.py first."
        )
        return 1

    legal_records = load_jsonl(LEGAL_INPUT_PATH)
    faq_records = load_jsonl(FAQ_INPUT_PATH)

    legal_chunks = chunk_legal_records(legal_records)
    faq_chunks = chunk_faq_records(faq_records)
    merged_chunks = legal_chunks + faq_chunks

    write_jsonl(LEGAL_OUTPUT_PATH, legal_chunks)
    write_jsonl(FAQ_OUTPUT_PATH, faq_chunks)
    write_jsonl(MERGED_OUTPUT_PATH, merged_chunks)

    print(
        f"Saved {len(legal_chunks)} legal chunks, {len(faq_chunks)} FAQ chunks, "
        f"and {len(merged_chunks)} merged chunks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
