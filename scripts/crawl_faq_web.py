import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_BASE_DIR = ROOT_DIR / "data" / "raw" / "faq_web"
RAW_PAGES_DIR = RAW_BASE_DIR / "pages"
RAW_MANIFEST_PATH = RAW_BASE_DIR / "manifest.jsonl"

DEFAULT_URL = "https://pbgdplthainguyen.gov.vn/index.php/faq/Luat-Dat-dai-nam-2024/"
DEFAULT_LOCAL_HTML = ROOT_DIR / "data" / "faq_web" / "pages" / (
    "Các câu hỏi thường gặp - Luật Đất đai năm 2024.html"
)
DEFAULT_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def compact_whitespace(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "faq-web"


def extract_first(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    return compact_whitespace(unescape(match.group(1)))


def fetch_url(url: str, timeout: int) -> Tuple[str, int]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace"), response.status


def save_html(raw_html: str, source_url: str, page_title: Optional[str]) -> Path:
    ensure_dir(RAW_PAGES_DIR)
    parsed = urlparse(source_url)
    base_name = page_title or parsed.path.rstrip("/").split("/")[-1] or "faq_page"
    file_name = f"{slugify(base_name)}.html"
    output_path = RAW_PAGES_DIR / file_name
    output_path.write_text(raw_html, encoding="utf-8")
    return output_path


def append_manifest(record: Dict) -> None:
    ensure_dir(RAW_MANIFEST_PATH.parent)
    with RAW_MANIFEST_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    return {
        "title": title,
        "source_url": canonical_url or og_url or fallback_url,
    }


def build_manifest_record(
    source_url: Optional[str],
    source_path: Path,
    title: Optional[str],
    input_type: str,
    status: str,
) -> Dict:
    return {
        "source_id": f"faq_{source_path.stem}",
        "source_type": "faq",
        "input_type": input_type,
        "title": title,
        "local_path": str(source_path.relative_to(ROOT_DIR)),
        "source_url": source_url,
        "language": "vi",
        "collected_at": now_iso(),
        "checksum": f"sha256:{hashlib.sha256(source_path.read_bytes()).hexdigest()}",
        "status": status,
    }


def crawl_from_web(url: str, timeout: int) -> Path:
    raw_html, http_status = fetch_url(url, timeout)
    source_meta = extract_page_metadata(raw_html, url)
    source_path = save_html(raw_html, source_meta.get("source_url") or url, source_meta.get("title"))
    append_manifest(
        build_manifest_record(
            source_url=source_meta.get("source_url") or url,
            source_path=source_path,
            title=source_meta.get("title"),
            input_type="html",
            status=str(http_status),
        )
    )
    return source_path


def register_local_html(path: Path, fallback_url: Optional[str]) -> Path:
    raw_html = path.read_text(encoding="utf-8", errors="replace")
    source_meta = extract_page_metadata(raw_html, fallback_url)
    append_manifest(
        build_manifest_record(
            source_url=source_meta.get("source_url") or fallback_url,
            source_path=path,
            title=source_meta.get("title"),
            input_type="html",
            status="local",
        )
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch FAQ HTML and save it as raw input for the RAG pipeline."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="FAQ page URL to fetch.")
    parser.add_argument(
        "--from-local",
        action="store_true",
        help="Register an existing local HTML file into the raw manifest instead of fetching.",
    )
    parser.add_argument(
        "--html-path",
        type=Path,
        default=DEFAULT_LOCAL_HTML,
        help="Path to a local HTML file used with --from-local.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Network timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.from_local:
            source_path = register_local_html(args.html_path, args.url)
            action = "registered"
        else:
            source_path = crawl_from_web(args.url, args.timeout)
            action = "saved"
    except FileNotFoundError as exc:
        print(f"Local HTML file not found: {exc}")
        return 1
    except HTTPError as exc:
        print(f"HTTP error while fetching FAQ page: {exc.code} {exc.reason}")
        return 1
    except URLError as exc:
        print(f"Network error while fetching FAQ page: {exc.reason}")
        return 1

    print(
        f"Successfully {action} raw FAQ HTML at {source_path.relative_to(ROOT_DIR)}. "
        "Run scripts/clean_faq_web.py to normalize it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
