import os
from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import HfApi, login

# --- CẤU HÌNH ---
# Thay đổi tên repository của bạn ở đây
REPO_NAME = "vn-land-law-2024" 
# Bạn có thể truyền token trực tiếp hoặc chạy `huggingface-cli login`
HF_TOKEN = os.getenv("HF_TOKEN") 

DATA_DIR = "/Users/huyhuy/Documents/HK2_2526/NLP/rag-chatbot/data/cleaned"

FILES = {
    "instruction": "phi3_finetune.jsonl",
    "qa_train": "train_qa.jsonl",
    "qa_test": "test_qa.jsonl",
    "corpus": "legal_docs.jsonl"
}

def upload_datasets():
    print(f"🚀 Bắt đầu quá trình chuẩn bị dữ liệu...")
    
    if not HF_TOKEN:
        print("⚠️ Cảnh báo: Không tìm thấy HF_TOKEN trong biến môi trường.")
        print("Vui lòng đảm bảo bạn đã đăng nhập bằng `huggingface-cli login` hoặc thiết lập biến HF_TOKEN.")

    try:
        # 1. Load Instruction Dataset (Fine-tuning)
        print(f"📦 Đang load {FILES['instruction']}...")
        ds_instruction = load_dataset("json", data_files=os.path.join(DATA_DIR, FILES['instruction']), split="train")
        
        # 2. Load QA Dataset (RAG Benchmark)
        print(f"📦 Đang load {FILES['qa_train']} và {FILES['qa_test']}...")
        ds_qa = load_dataset("json", data_files={
            "train": os.path.join(DATA_DIR, FILES['qa_train']),
            "test": os.path.join(DATA_DIR, FILES['qa_test'])
        })
        
        # 3. Load Corpus (Knowledge Base)
        print(f"📦 Đang load {FILES['corpus']}...")
        ds_corpus = load_dataset("json", data_files=os.path.join(DATA_DIR, FILES['corpus']), split="train")

        # 4. Push to Hub
        # Chúng ta sẽ push từng subset một để dễ quản lý
        
        print(f"📤 Đang đẩy subset 'instruction' lên {REPO_NAME}...")
        ds_instruction.push_to_hub(REPO_NAME, config_name="instruction", token=HF_TOKEN)
        
        print(f"📤 Đang đẩy subset 'rag_qa' lên {REPO_NAME}...")
        ds_qa.push_to_hub(REPO_NAME, config_name="rag_qa", token=HF_TOKEN)
        
        print(f"📤 Đang đẩy subset 'corpus' lên {REPO_NAME}...")
        ds_corpus.push_to_hub(REPO_NAME, config_name="corpus", token=HF_TOKEN)

        print(f"✅ Hoàn thành! Dataset của bạn đã sẵn sàng tại: https://huggingface.co/datasets/your-username/{REPO_NAME}")
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")

if __name__ == "__main__":
    upload_datasets()
