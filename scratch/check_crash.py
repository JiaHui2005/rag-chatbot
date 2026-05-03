import sys
import os

def test_imports():
    try:
        print("1. Kiem tra streamlit...")
        import streamlit as st
        print("   OK")
        
        print("2. Kiem tra yaml...")
        import yaml
        print("   OK")
        
        print("3. Kiem tra torch...")
        import torch
        print(f"   OK (Version: {torch.__version__})")
        
        print("4. Kiem tra transformers...")
        import transformers
        print(f"   OK (Version: {transformers.__version__})")
        
        print("5. Kiem tra chromadb...")
        import chromadb
        print("   OK")
        
        print("6. Kiem tra HuggingFaceEmbeddings...")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("   OK")
        
        print("7. Thu load Model Embedding (Buoc nay de bi crash nhat)...")
        # Thay model name bang model ban dang dung trong .env
        model_name = "YuITC/vietnamese-embedding-vn-legal"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        print("   OK - Load model thanh cong!")
        
    except Exception as e:
        print(f"\n[!] Phat hien loi: {str(e)}")

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    test_imports()
    print("\n[THANH CONG] Khong co thu vien nao bi crash memory.")
