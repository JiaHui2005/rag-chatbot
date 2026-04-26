import json
import re
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
EXTRACTED_TEXT_PATH = ROOT_DIR / "data" / "cleaned" / "legal_docs_extracted.jsonl"
CLEANED_OUTPUT_PATH = ROOT_DIR / "data" / "cleaned" / "legal_docs.jsonl"


ARTICLE_SPLIT_PATTERN = re.compile(
    r"(?=Điều\s+\d+[a-zA-Z]?\.\s+)",
    flags=re.MULTILINE,
)
ARTICLE_HEADER_PATTERN = re.compile(
    r"^Điều\s+(\d+[a-zA-Z]?)\.\s*(.+)$",
    flags=re.MULTILINE,
)
CHAPTER_PATTERN = re.compile(
    r"Chương\s+([IVXLC]+)\s*\n([^\n]+)",
    flags=re.IGNORECASE,
)


def compact_whitespace(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_law_text(text: str) -> str:
    text = compact_whitespace(text)
    text = re.sub(r"\n(Mục\s+\d+\.)", r"\n\n\1", text)
    text = re.sub(r"\n(Chương\s+[IVXLC]+)", r"\n\n\1", text)
    text = re.sub(r"\n(Điều\s+\d+[a-zA-Z]?\.)", r"\n\n\1", text)
    return text


def chapter_positions(text: str) -> List[Dict]:
    chapters: List[Dict] = []
    for match in CHAPTER_PATTERN.finditer(text):
        roman = match.group(1).upper()
        chapter_name = compact_whitespace(match.group(2))
        chapters.append(
            {
                "position": match.start(),
                "chapter": f"Chương {roman}",
                "chapter_title": chapter_name,
            }
        )
    return chapters


def chapter_for_position(chapters: List[Dict], position: int) -> Dict:
    current = {"chapter": None, "chapter_title": None}
    for chapter in chapters:
        if chapter["position"] <= position:
            current = chapter
        else:
            break
    return current


def infer_topic(article_title: str) -> str:
    slug = article_title.lower()
    slug = re.sub(r"[^0-9a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+", "_", slug)
    return slug.strip("_") or "general"


def split_articles(record: Dict) -> List[Dict]:
    text = normalize_law_text(record["content"])
    chapters = chapter_positions(text)
    pieces = [piece.strip() for piece in ARTICLE_SPLIT_PATTERN.split(text) if piece.strip()]
    documents: List[Dict] = []

    for piece in pieces:
        match = ARTICLE_HEADER_PATTERN.search(piece)
        if not match:
            continue

        article_number = match.group(1)
        article_title = compact_whitespace(match.group(2))
        article_position = text.find(piece)
        chapter = chapter_for_position(chapters, article_position)
        article_id = f"law_land_2024_article_{article_number.lower()}"
        article_content = compact_whitespace(piece)

        documents.append(
            {
                "doc_id": article_id,
                "source_id": record["source_id"],
                "source_type": "legal_document",
                "title": record["title"],
                "content": article_content,
                "metadata": {
                    "language": "vi",
                    "domain": "land_law",
                    "document_name": record["metadata"]["document_name"],
                    "document_type": "law",
                    "issued_year": record["metadata"]["issued_year"],
                    "effective_date": "2025-01-01",
                    "chapter": chapter["chapter"],
                    "chapter_title": chapter["chapter_title"],
                    "article": f"Điều {article_number}",
                    "clause": None,
                    "point": None,
                    "topic": infer_topic(article_title),
                    "source_url": record["metadata"]["source_url"],
                    "source_path": record["metadata"]["source_path"],
                    "collected_at": record["metadata"]["collected_at"],
                    "authority_level": "primary",
                    "tags": record["metadata"]["tags"],
                },
            }
        )

    return documents


def main() -> int:
    if not EXTRACTED_TEXT_PATH.exists():
        print("Missing extracted legal text. Run scripts/load_legal_docs.py first.")
        return 1

    records = load_jsonl(EXTRACTED_TEXT_PATH)
    cleaned_records: List[Dict] = []
    for record in records:
        cleaned_records.extend(split_articles(record))

    if not cleaned_records:
        print("No legal articles were extracted from the normalized text.")
        return 1

    write_jsonl(CLEANED_OUTPUT_PATH, cleaned_records)
    print(
        f"Saved {len(cleaned_records)} cleaned legal records to "
        f"{CLEANED_OUTPUT_PATH.relative_to(ROOT_DIR)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
