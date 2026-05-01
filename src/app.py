import streamlit as st
import time
import os
from dotenv import load_dotenv
import sys

# Đảm bảo import được core
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

@st.cache_resource
def load_engine():
    # Khởi tạo RAGEngine
    # Dùng cache để tránh load lại model 4GB mỗi lần bấm nút
    config = {"rag": {"chunk_size": 1000, "chunk_overlap": 200}}
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
        with st.spinner("Đang tải Mô hình (Phi-3) và Vector DB... Quá trình này có thể mất vài phút."):
            engine = load_engine()
        st.success("Hệ thống đã sẵn sàng!")
        st.info("Giao diện này phục vụ việc chạy thực nghiệm so sánh chất lượng câu trả lời giữa 4 cấu hình của đồ án.")

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
