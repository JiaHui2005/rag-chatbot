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

## Khởi chạy
- Mac/Linux: `./run.sh`
- Windows: `run.bat`
