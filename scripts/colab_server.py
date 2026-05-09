# --- COPY TOÀN BỘ CODE NÀY VÀO 1 CELL TRÊN GOOGLE COLAB ---

# 1. Cài đặt các thư viện cần thiết
# !pip install -U \
# fastapi \
# uvicorn \
# pyngrok \
# nest-asyncio \
# "transformers>=4.41.0" \
# "accelerate>=0.30.0" \
# "peft>=0.11.0" \
# "bitsandbytes>=0.43.0"

import nest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
import torch
import uvicorn
import nest_asyncio
from pyngrok import ngrok
from google.colab import drive
drive.mount('/content/drive')

# --- CẤU HÌNH ---
# Thay 'YOUR_NGROK_AUTH_TOKEN' bằng token của bạn từ https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "3D9KbBIP0gHzjlNfeDFuSMUozF7_trnfWZHmF1bNkppS4mcr" 
MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
ADAPTER_PATH = "/content/drive/MyDrive/phi3" # Đường dẫn đến thư mục phi3 trên Google Drive

# --- KHỞI TẠO APP ---
app = FastAPI()
nest_asyncio.apply()

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.3
    repetition_penalty: float = 1.1
    mode: str = "ft"

# --- LOAD MODEL ---
print("Đang load mô hình... Vui lòng đợi (có thể mất 2-5 phút)")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Nạp Tokenizer với cấu hình chuẩn cho Phi-3
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Nạp config và xử lý rope_scaling (Quan trọng để không bị lỗi KeyError: 'type')
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
    rope = config.rope_scaling
    # Nếu config bị thiếu trường 'type' (lỗi của remote code), override lại
    if isinstance(rope, dict) and "type" not in rope:
        print("⚠️ Đang tự động fix lỗi rope_scaling cho Phi-3...")
        config.rope_scaling = {
            "type": "longrope",
            "short_factor": 1.0,
            "long_factor": 1.0
        }

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Thử nạp adapter
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Đã load adapter thành công!")
except Exception as e:
    print(f"Cảnh báo: Không load được adapter ({e}). Chỉ sử dụng Base Model.")
    model = base_model

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        # Xử lý mode (bật/tắt adapter)
        if isinstance(model, PeftModel):
            if "base" in request.mode:
                try:
                    model.base_model.disable_adapter_layers() # Tắt adapter triệt để
                except:
                    pass
            else:
                try:
                    model.base_model.enable_adapter_layers() # Bật lại adapter
                except:
                    pass

        # Phi-3 yêu cầu định dạng Chat Template (<|user|>, <|assistant|>)
        # Nếu truyền raw text, model sẽ bị "ảo giác" (gibberish)
        messages = [
            {"role": "user", "content": request.prompt}
        ]
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tạo text và attention mask
        inputs = tokenizer(prompt_formatted, return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature, # Thường là 0.3
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=request.repetition_penalty,
                use_cache=True
            )
        
        # Decode và lấy phần trả lời (bỏ phần prompt ban đầu)
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
        return {"response": response_text.strip()}
    
    except Exception as e:
        print(f"LỖI SERVER: {str(e)}")
        return {"response": f"Lỗi phía Server Colab: {str(e)}"}

nest_asyncio.apply()

# Mở tunnel ngrok
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(8000).public_url

print(f"\n{'='*50}")
print(f"API PUBLIC URL: {public_url}")
print(f"Copy link này vào .env: COLAB_API_URL={public_url}")
print(f"{'='*50}\n")

# Chạy server theo cách async-safe cho Colab
config = uvicorn.Config(app, host="0.0.0.0", port=8000)
server = uvicorn.Server(config)
await server.serve()