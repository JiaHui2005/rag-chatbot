#!/bin/bash

# Kiểm tra môi trường ảo
if [ ! -d "venv" ]; then
    echo "Đang khởi tạo môi trường ảo..."
    python3 -m venv venv
fi

# Kích hoạt môi trường ảo
source venv/bin/activate

# Cài đặt dependencies
echo "Đang cài đặt thư viện..."
pip install -r requirements.txt

# Chạy ứng dụng Streamlit
echo "Đang khởi chạy Chatbot..."
streamlit run src/app.py
