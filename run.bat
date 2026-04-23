@echo off

:: Kiểm tra môi trường ảo
if not exist "venv" (
    echo Đang khởi tạo môi trường ảo...
    python -m venv venv
)

:: Kích hoạt môi trường ảo
call venv\Scripts\activate

:: Cài đặt dependencies
echo Đang cài đặt thư viện...
pip install -r requirements.txt

:: Chạy ứng dụng Streamlit
echo Đang khởi chạy Chatbot...
streamlit run src/app.py
