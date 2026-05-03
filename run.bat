@echo off
setlocal

:: Kiểm tra Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Python. Vui long cai dat Python va add vao PATH.
    pause
    exit /b
)

:: Kiểm tra môi trường ảo
if not exist "venv" (
    echo [INFO] Dang khoi tao moi truong ao...
    python -m venv venv
)

:: Kích hoạt môi trường ảo
echo [INFO] Dang kich hoat moi truong ao...
call venv\Scripts\activate

:: Cài đặt dependencies
echo [INFO] Dang kiem tra va cai dat thu vien (co the mat vai phut)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [LOI] Co loi khi cai dat thu vien. Vui long kiem tra ket noi mang.
    pause
    exit /b
)

:: Chạy ứng dụng Streamlit
echo [INFO] Dang khoi chay Chatbot...
:: Su dung python -m streamlit de dam bao dung moi truong
python -m streamlit run src/app.py --logger.level=debug --server.fileWatcherType none

if %errorlevel% neq 0 (
    echo.
    echo [LOI] Streamlit da dung dot ngot voi ma loi: %errorlevel%
)

echo.
echo [INFO] Nhan phim bat ky de thoat...
pause
