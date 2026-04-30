import argparse
import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw" / "faq_web" / "pages"
LEGACY_RAW_DIR = ROOT_DIR / "data" / "faq_web" / "pages"
CLEANED_OUTPUT_PATH = ROOT_DIR / "data" / "cleaned" / "faq_web.jsonl"
DEFAULT_URL = "https://pbgdplthainguyen.gov.vn/index.php/faq/Luat-Dat-dai-nam-2024/"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def compact_whitespace(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def strip_tags(html_fragment: str) -> str:
    # Remove script and style elements and their contents
    fragment = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html_fragment, flags=re.IGNORECASE | re.DOTALL)
    
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"</p\s*>", "\n", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"</div\s*>", "\n", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"<li\b[^>]*>", "\n- ", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return compact_whitespace(unescape(fragment))


def extract_first(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    return compact_whitespace(unescape(match.group(1)))


def extract_page_metadata(raw_html: str, fallback_url: Optional[str]) -> Dict[str, Optional[str]]:
    title = extract_first(r"<title>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
    canonical_url = extract_first(
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\'](.*?)["\']',
        raw_html,
        re.IGNORECASE | re.DOTALL,
    )
    og_url = extract_first(
        r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\'](.*?)["\']',
        raw_html,
        re.IGNORECASE | re.DOTALL,
    )
    topic = extract_first(
        r'<li[^>]*itemtype="https://schema\.org/ListItem"[^>]*>.*?<span[^>]*itemprop="name">([^<]+)</span>\s*</a><i[^>]*content="3"',
        raw_html,
        re.IGNORECASE | re.DOTALL,
    )
    return {
        "title": title,
        "source_url": canonical_url or og_url or fallback_url,
        "topic": topic or "Luật Đất đai năm 2024",
    }


def parse_faq_entries(raw_html: str, source_path: Path, source_meta: Dict[str, Optional[str]]) -> List[Dict]:
    # Strategy 1: Section-based (Thái Nguyên style)
    sections = re.findall(
        r'<div id="(sec\d+)" class="secclass">(.*?)</div>\s*</div>',
        raw_html,
        re.IGNORECASE | re.DOTALL,
    )
    
    records: List[Dict] = []
    if sections:
        for index, (section_id, section_html) in enumerate(sections):
            question = (
                extract_first(
                    r'<h3[^>]*class=["\'][^"\']*license-faqs__question[^"\']*["\'][^>]*>(.*?)</h3>',
                    section_html,
                    re.IGNORECASE | re.DOTALL,
                )
                or extract_first(r"<h2[^>]*>(.*?)</h2>", section_html, re.IGNORECASE | re.DOTALL)
            )
            answer_html = extract_first(
                r'<div class="answer">(.*)$',
                section_html,
                re.IGNORECASE | re.DOTALL,
            )
            if not question or not answer_html:
                continue

            answer = strip_tags(answer_html)
            answer = re.sub(r"^\s*Trả lời:\s*", "", answer, flags=re.IGNORECASE)
            answer = compact_whitespace(answer)
            if not answer:
                continue

            records.append(
                {
                    "doc_id": f"faq_{source_path.stem}_{index + 1:03d}",
                    "source_id": section_id,
                    "source_type": "faq",
                    "title": question,
                    "content": f"Câu hỏi: {question}\nTrả lời: {answer}",
                    "metadata": {
                        "language": "vi",
                        "domain": "land_law",
                        "question": question,
                        "answer": answer,
                        "topic": source_meta["topic"],
                        "source_url": source_meta["source_url"],
                        "source_path": str(source_path.relative_to(ROOT_DIR)),
                        "collected_at": now_iso(),
                        "authority_level": "secondary",
                        "tags": ["faq", "luat-dat-dai-2024"],
                    },
                }
            )
        return records

    # Strategy 2: Text-based splitting (Hưng Yên / General style)
    # Looking for markers like "Câu 1:", "Câu 1.", "Hỏi:", "Trả lời:"
    text = strip_tags(raw_html)
    
    # Split by "Câu \d+[:.]"
    parts = re.split(r"(Câu\s+\d+[\s.:]+)", text, flags=re.IGNORECASE)
    
    # parts[0] is usually preamble
    for i in range(1, len(parts), 2):
        marker = parts[i]
        content = parts[i+1] if i+1 < len(parts) else ""
        
        # Split content into question and answer
        # Common markers: "Hỏi:", "Trả lời:", or just the first paragraph
        sub_parts = re.split(r"(Trả\s+lời[\s:]+)", content, flags=re.IGNORECASE, maxsplit=1)
        
        if len(sub_parts) >= 3:
            question = compact_whitespace(sub_parts[0]).strip("Hỏi:").strip()
            answer = compact_whitespace(sub_parts[2])
        else:
            # Fallback: if no "Trả lời", try splitting by newlines
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            if not lines:
                continue
            question = lines[0]
            answer = "\n".join(lines[1:])
            
        if not question or not answer:
            continue
            
        records.append({
            "doc_id": f"faq_{source_path.stem}_{((i-1)//2) + 1:03d}",
            "source_id": marker.strip(),
            "source_type": "faq",
            "title": question,
            "content": f"Câu hỏi: {question}\nTrả lời: {answer}",
            "metadata": {
                "language": "vi",
                "domain": "land_law",
                "question": question,
                "answer": answer,
                "topic": source_meta["topic"],
                "source_url": source_meta["source_url"],
                "source_path": str(source_path.relative_to(ROOT_DIR)),
                "collected_at": now_iso(),
                "authority_level": "secondary",
                "tags": ["faq", "luat-dat-dai-2024"],
            },
        })
        
    return records


def deduplicate_records(records: List[Dict]) -> List[Dict]:
    seen_content = set()
    unique_records = []
    
    for record in records:
        # Normalize content for comparison
        content_norm = re.sub(r"\s+", " ", record["content"].lower()).strip()
        if content_norm not in seen_content:
            seen_content.add(content_norm)
            unique_records.append(record)
            
    return unique_records


def load_source_paths(explicit_path: Optional[Path]) -> List[Path]:
    if explicit_path:
        return [explicit_path]

    paths = sorted(RAW_DIR.glob("*.html"))
    if paths:
        return paths
    return sorted(LEGACY_RAW_DIR.glob("*.html"))


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw FAQ HTML into normalized JSONL records for RAG."
    )
    parser.add_argument(
        "--html-path",
        type=Path,
        help="Optional path to a single raw FAQ HTML file to clean.",
    )
    parser.add_argument(
        "--fallback-url",
        default=DEFAULT_URL,
        help="Fallback URL if the page does not expose a canonical source URL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_paths = load_source_paths(args.html_path)
    if not source_paths:
        print("No raw FAQ HTML files found. Run scripts/crawl_faq_web.py first.")
        return 1

    all_records: List[Dict] = []
    for source_path in source_paths:
        raw_html = source_path.read_text(encoding="utf-8", errors="replace")
        source_meta = extract_page_metadata(raw_html, args.fallback_url)
        all_records.extend(parse_faq_entries(raw_html, source_path, source_meta))

    if not all_records:
        print("No FAQ entries were extracted. Please inspect the page structure.")
        return 1

    # Deduplicate
    initial_count = len(all_records)
    all_records = deduplicate_records(all_records)
    print(f"Deduplicated: {initial_count} -> {len(all_records)} records.")

    write_jsonl(CLEANED_OUTPUT_PATH, all_records)
    print(
        f"Saved {len(all_records)} FAQ records to "
        f"{CLEANED_OUTPUT_PATH.relative_to(ROOT_DIR)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
