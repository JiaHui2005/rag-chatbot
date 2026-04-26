import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_LEGAL_DIR = ROOT_DIR / "data" / "raw" / "legal_docs"
RAW_MANIFEST_PATH = RAW_LEGAL_DIR / "manifest.jsonl"
EXTRACTED_DIR = ROOT_DIR / "data" / "cleaned"
EXTRACTED_TEXT_PATH = EXTRACTED_DIR / "legal_docs_extracted.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compact_whitespace(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = value.replace("\ufeff", " ")
    value = value.replace("\u2028", "\n")
    value = value.replace("\u2029", "\n")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in value.split("\n")]
    filtered = "\n".join(line for line in lines if line)
    return filtered.strip()


def sha256_for_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path: Path, records: Iterable[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_pdf(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyPDF2 is not installed in the current Python environment."
        ) from exc

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return compact_whitespace("\n\n".join(pages))


def read_doc_with_textutil(path: Path) -> str:
    textutil = shutil.which("textutil")
    if not textutil:
        raise RuntimeError("textutil is required to extract .doc files on this machine.")

    result = subprocess.run(
        [textutil, "-convert", "txt", "-stdout", str(path)],
        check=True,
        capture_output=True,
    )
    return compact_whitespace(result.stdout.decode("utf-8", errors="replace"))


def detect_document_name(text: str, default_name: str) -> str:
    for line in text.splitlines()[:20]:
        stripped = line.strip()
        if stripped.upper() == "ĐẤT ĐAI":
            return "Luật Đất đai 2024"
        if "LUẬT ĐẤT ĐAI" in stripped.upper():
            return "Luật Đất đai 2024"
    return default_name


def build_record(path: Path, extracted_text: str) -> Dict:
    input_type = path.suffix.lower().lstrip(".")
    title = detect_document_name(extracted_text, path.stem)
    return {
        "source_id": f"legal_{path.stem}",
        "source_type": "legal_document",
        "input_type": input_type,
        "title": title,
        "content": extracted_text,
        "metadata": {
            "language": "vi",
            "domain": "land_law",
            "document_name": title,
            "document_type": "law",
            "issued_year": 2024,
            "source_url": None,
            "source_path": str(path.relative_to(ROOT_DIR)),
            "collected_at": now_iso(),
            "authority_level": "primary",
            "checksum": f"sha256:{sha256_for_file(path)}",
            "tags": ["luat-dat-dai", "legal-document"],
        },
    }


def build_manifest_record(path: Path, title: str) -> Dict:
    return {
        "source_id": f"legal_{path.stem}",
        "source_type": "legal_document",
        "input_type": path.suffix.lower().lstrip("."),
        "title": title,
        "local_path": str(path.relative_to(ROOT_DIR)),
        "source_url": None,
        "language": "vi",
        "collected_at": now_iso(),
        "checksum": f"sha256:{sha256_for_file(path)}",
    }


def gather_input_files() -> List[Path]:
    paths: List[Path] = []
    for pattern in ("pdf/*.pdf", "doc/*.doc", "doc/*.docx"):
        paths.extend(sorted(RAW_LEGAL_DIR.glob(pattern)))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract raw legal documents into text JSONL.")
    parser.add_argument(
        "--prefer",
        choices=["pdf", "doc", "all"],
        default="doc",
        help="Prefer one source type if both PDF and DOC exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_files = gather_input_files()
    if not input_files:
        print("No legal document files found under data/raw/legal_docs.")
        return 1

    manifest_records: List[Dict] = []
    extracted_records: List[Dict] = []

    for path in input_files:
        suffix = path.suffix.lower().lstrip(".")
        if args.prefer != "all" and suffix != args.prefer:
            continue

        if suffix == "pdf":
            extracted_text = read_pdf(path)
        elif suffix in {"doc", "docx"}:
            extracted_text = read_doc_with_textutil(path)
        else:
            continue

        if not extracted_text:
            continue

        record = build_record(path, extracted_text)
        extracted_records.append(record)
        manifest_records.append(build_manifest_record(path, record["title"]))

    if not extracted_records:
        print("No legal documents were extracted.")
        return 1

    append_jsonl(RAW_MANIFEST_PATH, manifest_records)
    append_jsonl(EXTRACTED_TEXT_PATH, extracted_records)
    print(
        f"Saved {len(extracted_records)} extracted legal document records to "
        f"{EXTRACTED_TEXT_PATH.relative_to(ROOT_DIR)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
