import streamlit as st
import time
import os
from pathlib import Path
from dotenv import load_dotenv
import sys
import yaml

# Đảm bảo import được core
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

@st.cache_resource
def load_engine():
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return RAGEngine(config)

def init_page():
    st.set_page_config(
        page_title="RAG AI - So sánh cấu hình",
        page_icon="🤖",
        layout="wide"
    )
    st.header("🤖 Hệ thống RAG - Đánh giá 4 Cấu hình")
    st.markdown("---")

def main():
    init_page()
    
    with st.sidebar:
        st.title("⚙️ Trạng thái Hệ thống")
        with st.spinner("Đang tải Mô hình..."):
            engine = load_engine()
        st.success("Hệ thống đã sẵn sàng!")
        
        st.markdown("---")
        st.subheader("📁 Quản lý Tài liệu (RAG)")
        uploaded_files = st.file_uploader(
            "Tải lên tài liệu mới (PDF, TXT)", 
            accept_multiple_files=True,
            type=["pdf", "txt"]
        )
        
        if st.button("🛠️ Xử lý và Nạp vào DB", use_container_width=True):
            if uploaded_files:
                with st.spinner("Đang xử lý tài liệu..."):
                    # Lưu file tạm và lấy đường dẫn
                    file_paths = []
                    os.makedirs("data", exist_ok=True)
                    for uploaded_file in uploaded_files:
                        path = os.path.join("data", uploaded_file.name)
                        with open(path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(path)
                    
                    # Gọi engine để xử lý
                    engine.process_documents(file_paths)
                    st.success(f"Đã nạp xong {len(file_paths)} tài liệu!")
                    st.rerun()
            else:
                st.warning("Vui lòng chọn file trước!")
        
        if st.button("🔄 Nạp lại từ thư mục 'knowledge_base'", use_container_width=True):
            data_dir = os.path.join("data", "knowledge_base")
            if os.path.exists(data_dir):
                import glob
                # Tìm tất cả file tài liệu trong data và các thư mục con
                extensions = ["*.pdf", "*.txt", "*.md", "*.json", "*.jsonl", "*.docx", "*.html"]
                files = []
                for ext in extensions:
                    files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
                # Bỏ qua các file tạm của Word (bắt đầu bằng ~$) hoặc file ẩn
                files = [f for f in files if not os.path.basename(f).startswith("~$") and not os.path.basename(f).startswith(".")]
                
                if files:
                    with st.spinner(f"Đang xử lý {len(files)} tài liệu tìm thấy..."):
                        try:
                            engine.process_documents(files)
                            st.success(f"Đã nạp xong {len(files)} tài liệu!")
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Đã xảy ra lỗi khi nạp tài liệu: {str(e)}")
                else:
                    st.warning("Không tìm thấy file .pdf, .txt, .md, .docx, .html nào trong thư mục 'knowledge_base'!")

        if st.button("🗑️ Xóa toàn bộ Database", use_container_width=True):
            if os.path.exists("./db"):
                try:
                    # Giải phóng bộ nhớ trong Streamlit và Chroma để nhả lock file
                    st.cache_resource.clear()
                    if hasattr(engine, 'vector_db'):
                        engine.vector_db = None
                    try:
                        import chromadb
                        chromadb.api.client.SharedSystemClient.clear_system_cache()
                    except:
                        pass
                    import gc
                    gc.collect()
                    time.sleep(0.5)
                    
                    import shutil
                    shutil.rmtree("./db")
                    st.success("Đã xóa Database cũ. Vui lòng nạp lại tài liệu.")
                    st.rerun()
                except PermissionError:
                    st.error("❌ Lỗi: Database đang bị hệ thống khóa. Bạn hãy vào Terminal (nơi đang chạy lệnh streamlit), nhấn `Ctrl + C` để tắt app, sau đó chạy lại lệnh `streamlit run src/app.py` rồi hãy nhấn nút Xóa này nhé!")
                except Exception as e:
                    st.error(f"Lỗi không xác định khi xóa: {e}")
        
        st.markdown("---")
        st.info("Giao diện này phục vụ việc chạy thực nghiệm so sánh chất lượng câu trả lời giữa 4 cấu hình.")

    st.markdown("### ❓ Nhập câu hỏi của bạn")
    question = st.text_input("Ví dụ: Điều kiện để được cấp sổ đỏ năm 2024 là gì?", key="user_question")
    
    if st.button("🚀 Gửi câu hỏi và So sánh", type="primary"):
        if not question.strip():
            st.warning("Vui lòng nhập câu hỏi!")
            return
            
        st.markdown("---")
        st.markdown(f"**Câu hỏi:** {question}")
        
        # Tạo lưới 2x2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cấu hình A: Base LLM (Không RAG)")
            with st.spinner("Đang tạo câu trả lời..."):
                start = time.time()
                ans_a = engine.query(question, mode="base")
                end = time.time()
            st.info(ans_a)
            st.caption(f"⏱️ Thời gian: {end-start:.2f}s")
            
        with col2:
            st.subheader("Cấu hình C: Fine-Tuned LLM (Không RAG)")
            with st.spinner("Đang tạo câu trả lời..."):
                start = time.time()
                ans_c = engine.query(question, mode="ft")
                end = time.time()
            st.success(ans_c)
            st.caption(f"⏱️ Thời gian: {end-start:.2f}s")
            
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Cấu hình B: Base LLM (+ RAG)")
            with st.spinner("Đang tạo câu trả lời..."):
                start = time.time()
                ans_b = engine.query(question, mode="base_rag")
                end = time.time()
            st.warning(ans_b)
            st.caption(f"⏱️ Thời gian: {end-start:.2f}s")
            
        with col4:
            st.subheader("Cấu hình D: Fine-Tuned LLM (+ RAG)")
            with st.spinner("Đang tạo câu trả lời..."):
                start = time.time()
                ans_d = engine.query(question, mode="ft_rag")
                end = time.time()
            st.error(ans_d)
            st.caption(f"⏱️ Thời gian: {end-start:.2f}s")

if __name__ == "__main__":
    main()
