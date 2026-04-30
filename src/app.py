import streamlit as st
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_page():
    st.set_page_config(
        page_title="RAG AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    st.header("🤖 RAG AI Assistant")
    st.markdown("---")

def init_sidebar():
    with st.sidebar:
        st.title("⚙️ Hệ thống")
        st.info("Trợ lý AI hỗ trợ tra cứu Luật Đất đai năm 2024 và các câu hỏi thường gặp.")
        
        st.markdown("---")
        if st.button("🗑️ Xóa lịch sử chat"):
            st.session_state.messages = []
            st.rerun()

def main():
    init_page()
    init_sidebar()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response (Mockup RAG)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Giả lập phản hồi từ RAG
            mock_response = f"Đây là phản hồi giả lập cho câu hỏi: '{prompt}'. Hệ thống RAG đang được cấu hình..."
            
            for chunk in mock_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
