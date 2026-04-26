# RAG Chatbot Project

Hệ thống Chatbot RAG (Retrieval-Augmented Generation) được xây dựng với Streamlit và LangChain.

## Cấu trúc dự án
- `data/`: Lưu trữ tài liệu thô (PDF, TXT, ...).
- `db/`: Lưu trữ Vector Database (ChromaDB).
- `src/`: Mã nguồn chính.
    - `core/`: Logic xử lý RAG, embeddings, và LLM.
    - `app.py`: Giao diện người dùng Streamlit.
- `config/`: Cấu hình hệ thống.
- `scripts/`: Script hỗ trợ.

## Cài đặt
1. Cài đặt Python 3.9+
2. Cài đặt thư viện: `pip install -r requirements.txt`
3. Cấu hình file `.env` từ `.env.example`.

## Pipeline dữ liệu RAG

Project hiện dùng pipeline 4 tầng:

1. `raw`: lưu nguồn gốc ban đầu
   - Luật: `data/raw/legal_docs/`
   - FAQ web: `data/raw/faq_web/pages/`
2. `cleaned`: dữ liệu đã chuẩn hóa về JSONL
   - `data/cleaned/legal_docs.jsonl`
   - `data/cleaned/faq_web.jsonl`
3. `processed`: dữ liệu đã chunk để embedding
   - `data/processed/merged_chunks.jsonl`
4. `db`: vector database Chroma
   - `db/chroma/`

## Chạy pipeline ingest và index

Thứ tự chạy khuyến nghị:

1. Crawl hoặc đăng ký HTML FAQ thô:
```bash
venv/bin/python scripts/crawl_faq_web.py --from-local
```

2. Làm sạch FAQ web:
```bash
venv/bin/python scripts/clean_faq_web.py
```

3. Trích xuất văn bản luật từ `PDF/DOC`:
```bash
venv/bin/python scripts/load_legal_docs.py
```

4. Chuẩn hóa và tách luật theo từng điều:
```bash
venv/bin/python scripts/clean_legal_docs.py
```

5. Chunk luật và FAQ thành dữ liệu cho embedding:
```bash
venv/bin/python scripts/build_chunks.py
```

6. Build Chroma index:
```bash
venv/bin/python scripts/build_index.py
```

Ghi chú:
- `crawl_faq_web.py` chỉ lo lưu HTML thô và manifest.
- `clean_faq_web.py` mới là bước parse HTML FAQ thành JSONL sạch.
- `build_index.py` sẽ thử dùng embedding model cấu hình trong `config/config.yaml`. Nếu môi trường không có mạng và chưa cache model, script sẽ fallback sang embedding offline đơn giản để bạn test pipeline end-to-end.

## Khởi chạy
- Mac/Linux: `./run.sh`
- Windows: `run.bat`
