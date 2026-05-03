import json
import os
import sys
import time
import pandas as pd
from pathlib import Path
import yaml
from dotenv import load_dotenv
import evaluate
import nltk
from tqdm import tqdm
import argparse

# Ensure core can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from core.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_engine():
    root_dir = Path(__file__).resolve().parents[1]
    config_path = root_dir / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return RAGEngine(config)

def evaluate_mode(engine, eval_data, mode, metrics, limit=None):
    """Đánh giá cho một cấu hình cụ thể."""
    rouge = metrics['rouge']
    bleu = metrics['bleu']
    bertscore = metrics['bertscore']
    
    results = []
    human_eval_column = []
    
    print(f"\n--- Đang đánh giá chế độ: {mode.upper()} ---")
    
    # Giới hạn số câu nếu có
    test_items = eval_data[:limit] if limit else eval_data
    
    for item in tqdm(test_items):
        query = item["question"]
        reference_answer = item["answer"]
        gt_doc_id = item.get("doc_id")
        
        # 1. Retrieval (Chỉ thực hiện nếu mode có RAG)
        recall_at_5 = 0
        retrieval_time = 0
        retrieved_ids = []
        context = ""
        
        if "_rag" in mode:
            start_retrieval = time.time()
            _, docs = engine.retrieve_documents(query)
            end_retrieval = time.time()
            retrieval_time = end_retrieval - start_retrieval
            retrieved_ids = [doc.metadata.get("chunk_id", "") for doc in docs]
            
            # Tính Recall@5 dựa trên doc_id gốc
            # Kiểm tra xem doc_id của câu hỏi có nằm trong metadata của 5 chunk đầu tiên không
            for doc in docs[:5]:
                if gt_doc_id and gt_doc_id in doc.metadata.get("doc_id", ""):
                    recall_at_5 = 1
                    break
            
            context = engine.build_context(docs[:3])
        
        # 2. Generation
        start_gen = time.time()
        # Chế độ query của engine: 'base', 'ft' (không RAG) hoặc 'base_rag', 'ft_rag' (có RAG)
        generated_answer = engine.query(query, mode=mode)
        end_gen = time.time()
        
        # 3. Calculate Metrics
        # BLEU
        try:
            b_score = bleu.compute(predictions=[generated_answer], references=[[reference_answer]])['bleu']
        except:
            b_score = 0.0
            
        # ROUGE
        try:
            r_results = rouge.compute(predictions=[generated_answer], references=[reference_answer])
            rouge_l = r_results['rougeL']
        except:
            rouge_l = 0.0
            
        # BERTScore
        try:
            bert_results = bertscore.compute(predictions=[generated_answer], references=[reference_answer], lang="vi")
            bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])
        except:
            bert_f1 = 0.0
            
        res = {
            "query": query,
            "recall_at_5": recall_at_5,
            "bleu": b_score,
            "rouge_l": rouge_l,
            "bert_f1": bert_f1,
            "retrieval_time": retrieval_time,
            "generation_time": end_gen - start_gen,
            "answer": generated_answer
        }
        results.append(res)
        human_eval_column.append(generated_answer)

    # Tính trung bình
    summary = {
        "mode": mode,
        "total_queries": len(results),
        "avg_recall_at_5": sum(r["recall_at_5"] for r in results) / len(results) if "_rag" in mode else None,
        "avg_bleu": sum(r["bleu"] for r in results) / len(results),
        "avg_rouge_l": sum(r["rouge_l"] for r in results) / len(results),
        "avg_bert_f1": sum(r["bert_f1"] for r in results) / len(results),
        "avg_generation_time": sum(r["generation_time"] for r in results) / len(results)
    }
    
    return summary, results, human_eval_column

def main():
    parser = argparse.ArgumentParser(description="Đánh giá hệ thống RAG với 4 cấu hình.")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số câu hỏi đánh giá (mặc định: toàn bộ 50 câu).")
    parser.add_argument("--modes", type=str, default="base,base_rag,ft,ft_rag", help="Các chế độ cần chạy, cách nhau bởi dấu phẩy.")
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parents[1]
    test_data_path = root_dir / "data" / "cleaned" / "test_qa.jsonl"
    output_dir = root_dir / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print(f"Đang tải dữ liệu test từ {test_data_path}...")
    eval_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Lấy thông tin từ metadata hoặc root tùy cấu hình file jsonl
            q = item.get("metadata", {}).get("question") or item.get("title")
            a = item.get("metadata", {}).get("answer") or item.get("content")
            eval_data.append({
                "question": q,
                "answer": a,
                "doc_id": item.get("doc_id")
            })
    
    if args.limit:
        eval_data = eval_data[:args.limit]
    print(f"Tổng số câu hỏi sẽ đánh giá: {len(eval_data)}")

    # 2. Load Metrics & Engine
    print("Đang khởi tạo Metrics và RAG Engine...")
    engine = load_engine()
    metrics = {
        'rouge': evaluate.load('rouge'),
        'bleu': evaluate.load('bleu'),
        'bertscore': evaluate.load('bertscore')
    }
    
    modes_to_run = args.modes.split(",")
    all_summaries = []
    
    # 3. Chạy đánh giá cho từng Mode
    for mode in modes_to_run:
        mode = mode.strip()
        summary, detailed_results, answers = evaluate_mode(engine, eval_data, mode, metrics)
        all_summaries.append(summary)
        
        # Lưu kết quả chi tiết của mode này để dùng cho bước tổng hợp sau
        with open(output_dir / f"results_{mode}.json", "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=4)

    # 4. Lưu tổng hợp và xây dựng file Human Eval (Dạng hàng dọc cho dễ chấm)
    with open(output_dir / "summary_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=4)
        
    print("Đang tổng hợp file Human Evaluation dạng hàng dọc...")
    
    all_answers = {} # {mode: [list_of_answers]}
    # Giả sử chúng ta đã thu thập answers trong vòng lặp ở bước 3
    # Ở bước 3 cũ, chúng ta đã lưu vào Answers_mode.
    # Để chắc chắn, mình sẽ map lại từ detailed_results nếu cần, 
    # nhưng ở đây mình sẽ chỉnh lại logic trong vòng lặp chính.

    human_eval_rows = []
    for mode in modes_to_run:
        m = mode.strip()
        # Lấy câu trả lời từ file results đã lưu của mode này
        with open(output_dir / f"results_{m}.json", "r", encoding="utf-8") as rf:
            results_data = json.load(rf)
            
        for i, item in enumerate(eval_data):
            q_id = i + 1
            question = item["question"]
            ground_truth = item["answer"]
            ans = results_data[i]["answer"]
            
            row = {
                "Model": m.upper(),
                "ID": q_id,
                "Question": question,
                "Ground Truth": ground_truth,
                "Generated Answer": ans,
                "Acc (1-5)": "",
                "Faith (1-5)": "",
                "Comp (1-5)": "",
                "Flu (1-5)": "",
                "Avg Score": "", 
                "Notes": ""
            }
            human_eval_rows.append(row)

    human_eval_df_vertical = pd.DataFrame(human_eval_rows)
    human_eval_path = output_dir / "human_eval_full_50.csv"
    
    # Xuất file CSV
    human_eval_df_vertical.to_csv(human_eval_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*50)
    print("ĐÁNH GIÁ HOÀN TẤT!")
    print(f"Báo cáo định lượng (JSON): {output_dir / 'summary_comparison.json'}")
    print(f"File chấm điểm Human Eval (Dạng hàng): {human_eval_path}")
    print("\nLƯU Ý:")
    print("1. File đã chia mỗi câu hỏi thành 4 dòng ứng với 4 mô hình (A, B, C, D).")
    print("2. Bạn hãy nhập điểm 1-5 vào các cột Acc, Faith, Comp, Flu.")
    print("3. Công thức tính trung bình: =AVERAGE(F2:I2)")
    print("="*50)
    
    # In bảng tóm tắt nhanh
    print("\nBảng so sánh kết quả định lượng (Auto Metrics):")
    print(f"{'Mode':<10} | {'Recall@5':<10} | {'BLEU':<10} | {'ROUGE-L':<10} | {'BERT-F1':<10}")
    for s in all_summaries:
        r5 = f"{s['avg_recall_at_5']:.4f}" if s['avg_recall_at_5'] is not None else "N/A"
        print(f"{s['mode']:<10} | {r5:<10} | {s['avg_bleu']:<10.4f} | {s['avg_rouge_l']:<10.4f} | {s['avg_bert_f1']:<10.4f}")

if __name__ == "__main__":
    main()
