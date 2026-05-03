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
- `build_index.py` đọc `rag.persist_dir`, `rag.collection_name`, `rag.embedding_backend` và `rag.embedding_model` trong `config/config.yaml`.
- Sau khi build, index ghi thêm `db/chroma/index_manifest.json`; app đọc manifest này để dùng đúng embedding backend khi truy vấn.
- Nếu môi trường không có mạng và chưa cache model HuggingFace, `embedding_backend: auto` sẽ fallback sang embedding offline đơn giản để test end-to-end. Khi muốn dùng embedding HuggingFace thật, hãy đảm bảo model đã tải được rồi chạy lại:
```bash
venv/bin/python scripts/build_index.py --embedding-backend huggingface
```
- Retrieval hiện kết hợp Chroma vector search + BM25 từ `data/processed/merged_chunks.jsonl`, sau đó rerank theo từ khóa, điều luật, metadata `article/topic`, và tự kéo thêm chunk liền kề cùng điều để tránh mất nội dung khi một điều bị tách nhiều chunk.

## Khởi chạy
- Mac/Linux: `./run.sh`
- Windows: `run.bat`
