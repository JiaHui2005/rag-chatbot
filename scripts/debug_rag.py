import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import faulthandler
faulthandler.enable()

import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Thêm src vào path để import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from core.rag_engine import RAGEngine

def debug_main():
    print("--- DEBUG RAG START ---")
    
    # 1. Load Config
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    print(f"Loading config from: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    
    # 2. Init Engine
    print("Initializing RAGEngine...")
    try:
        engine = RAGEngine(config)
        print("[OK] RAGEngine initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Error during RAGEngine init: {e}")
        return

    # 3. Test Retrieval
    question = "Điều kiện cấp sổ đỏ năm 2024"
    print(f"\nTesting Retrieval with question: '{question}'")
    try:
        query, docs = engine.retrieve_documents(question)
        print(f"[OK] Retrieval done. Found {len(docs)} documents.")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc.metadata.get('chunk_id')} - {doc.page_content[:100]}...")
    except Exception as e:
        print(f"[ERROR] Error during retrieval: {e}")

    # 4. Test Query (Remote LLM)
    print(f"\nTesting Query (mode='base_rag')...")
    try:
        answer = engine.query(question, mode="base_rag")
        print("\n--- ANSWER ---")
        print(answer)
        print("--------------")
    except Exception as e:
        print(f"[ERROR] Error during query: {e}")

    print("\n--- DEBUG RAG END ---")

if __name__ == "__main__":
    debug_main()
