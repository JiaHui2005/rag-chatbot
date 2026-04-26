# Scripts Directory
Thư mục này chứa các script hỗ trợ quá trình phát triển, bảo trì và triển khai dự án.

### Pipeline hiện tại

- `crawl_faq_web.py`: crawl hoặc đăng ký HTML FAQ thô vào `data/raw/faq_web/pages/` và ghi `manifest.jsonl`
- `clean_faq_web.py`: parse HTML FAQ, làm sạch `question/answer`, xuất `data/cleaned/faq_web.jsonl`
- `load_legal_docs.py`: đọc luật từ `PDF/DOC`, trích text gốc ra `data/cleaned/legal_docs_extracted.jsonl`
- `clean_legal_docs.py`: chuẩn hóa văn bản luật và tách thành từng `Điều`, xuất `data/cleaned/legal_docs.jsonl`
- `build_chunks.py`: chunk luật và FAQ thành `data/processed/*.jsonl`
- `build_index.py`: đưa `merged_chunks.jsonl` vào Chroma tại `db/chroma/`

### Thứ tự chạy

```bash
venv/bin/python scripts/crawl_faq_web.py --from-local
venv/bin/python scripts/clean_faq_web.py
venv/bin/python scripts/load_legal_docs.py
venv/bin/python scripts/clean_legal_docs.py
venv/bin/python scripts/build_chunks.py
venv/bin/python scripts/build_index.py
```
