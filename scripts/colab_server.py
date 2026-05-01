# --- COPY TOÀN BỘ CODE NÀY VÀO 1 CELL TRÊN GOOGLE COLAB ---

# 1. Cài đặt các thư viện cần thiết
# !pip install fastapi uvicorn pyngrok nest-asyncio transformers peft bitsandbytes accelerate

import nest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch
import uvicorn
from pyngrok import ngrok

# --- CẤU HÌNH ---
# Thay 'YOUR_NGROK_AUTH_TOKEN' bằng token của bạn từ https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN" 
MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
ADAPTER_PATH = "/content/drive/MyDrive/phi3" # Đường dẫn đến thư mục phi3 trên Google Drive

# --- KHỞI TẠO APP ---
app = FastAPI()
nest_asyncio.apply()

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.3
    mode: str = "ft"

# --- LOAD MODEL ---
print("Đang load mô hình... Vui lòng đợi (có thể mất 2-5 phút)")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# Thử load adapter nếu có
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Đã load adapter thành công!")
except Exception as e:
    print(f"Cảnh báo: Không load được adapter ({e}). Chỉ sử dụng Base Model.")
    model = base_model

@app.post("/generate")
async def generate(request: GenerateRequest):
    # Xử lý mode (bật/tắt adapter)
    if hasattr(model, "disable_adapter"):
        if "base" in request.mode:
            model.disable_adapter()
        else:
            model.enable_adapter()
    
    # Tạo text
    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Cắt bỏ phần prompt ban đầu trong kết quả trả về nếu cần
    if response_text.startswith(request.prompt):
        response_text = response_text[len(request.prompt):].strip()
        
    return {"response": response_text}

# --- CHẠY SERVER ---
if __name__ == "__main__":
    # Kết nối ngrok
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(8000).public_url
    print(f"\n\n{'='*50}")
    print(f"API PUBLIC URL: {public_url}")
    print(f"Hãy copy link trên vào file .env ở máy của bạn (biến COLAB_API_URL)")
    print(f"{'='*50}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
