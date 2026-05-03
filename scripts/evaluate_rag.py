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

def evaluate_rag():
    print("Initializing RAG Engine and Metrics...")
    engine = load_engine()
    
    # Load metrics
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    bertscore = evaluate.load('bertscore')
    
    root_dir = Path(__file__).resolve().parents[1]
    eval_set_path = root_dir / "data" / "evaluation" / "evaluation_set.json"
    
    if not eval_set_path.exists():
        print(f"Evaluation set not found at {eval_set_path}")
        return

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []
    human_eval_data = []

    print(f"Running evaluation on {len(eval_set)} queries...")
    
    for item in tqdm(eval_set):
        query = item["query"]
        ground_truth_context_id = item.get("ground_truth_context")
        reference_answer = item.get("reference_answer")
        
        # 1. Retrieval Evaluation
        start_retrieval = time.time()
        retrieval_query, docs = engine.retrieve_documents(query)
        end_retrieval = time.time()
        
        retrieved_chunk_ids = [doc.metadata.get("chunk_id") for doc in docs]
        recall_at_5 = 1 if ground_truth_context_id in retrieved_chunk_ids[:5] else 0
        
        # 2. Generation Evaluation
        start_gen = time.time()
        generated_answer = engine.query(query, mode="base_rag")
        end_gen = time.time()
        
        # Calculate Metrics
        # BLEU
        bleu_score = bleu.compute(predictions=[generated_answer], references=[[reference_answer]])['bleu']
        
        # ROUGE
        rouge_results = rouge.compute(predictions=[generated_answer], references=[reference_answer])
        rouge_l = rouge_results['rougeL']
        
        # BERTScore
        bert_results = bertscore.compute(predictions=[generated_answer], references=[reference_answer], lang="vi")
        bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])
        
        res = {
            "id": item["id"],
            "query": query,
            "recall_at_5": recall_at_5,
            "bleu": bleu_score,
            "rouge_l": rouge_l,
            "bert_f1": bert_f1,
            "retrieval_time": end_retrieval - start_retrieval,
            "generation_time": end_gen - start_gen,
            "retrieved_chunks": retrieved_chunk_ids
        }
        results.append(res)
        
        human_eval_data.append({
            "ID": item["id"],
            "Query": query,
            "Retrieved Context": engine.build_context(docs[:3]),
            "Generated Answer": generated_answer,
            "Ground Truth": reference_answer,
            "Score (1-5)": "",
            "Notes": ""
        })

    # Save results
    results_path = root_dir / "data" / "evaluation" / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Summary results
    summary = {
        "total_queries": len(results),
        "avg_recall_at_5": sum(r["recall_at_5"] for r in results) / len(results),
        "avg_bleu": sum(r["bleu"] for r in results) / len(results),
        "avg_rouge_l": sum(r["rouge_l"] for r in results) / len(results),
        "avg_bert_f1": sum(r["bert_f1"] for r in results) / len(results),
        "avg_generation_time": sum(r["generation_time"] for r in results) / len(results)
    }
    
    summary_path = root_dir / "data" / "evaluation" / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    # Save Human Evaluation Template
    human_eval_df = pd.DataFrame(human_eval_data)
    human_eval_path = root_dir / "data" / "evaluation" / "human_eval_template.csv"
    human_eval_df.to_csv(human_eval_path, index=False, encoding="utf-8-sig")

    print("\nEvaluation complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"Detailed results: {results_path}")
    print(f"Human eval template: {human_eval_path}")
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    evaluate_rag()
