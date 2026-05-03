import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    # 1. Load data
    root_dir = Path(__file__).resolve().parents[1]
    json_path = root_dir / "data" / "evaluation" / "summary_comparison.json"
    output_dir = root_dir / "data" / "evaluation" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"Error: Không tìm thấy file {json_path}. Hãy chạy script evaluate_rag.py trước.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # Đổi tên mode cho đẹp trong biểu đồ
    mode_map = {
        "base": "A: Base (No RAG)",
        "base_rag": "B: Base + RAG",
        "ft": "C: Fine-tuned",
        "ft_rag": "D: FT + RAG"
    }
    df['mode_name'] = df['mode'].map(mode_map)

    # 2. Chuẩn bị dữ liệu cho biểu đồ Metrics (BLEU, ROUGE, BERTScore)
    metrics_df = df.melt(id_vars='mode_name', 
                         value_vars=['avg_bleu', 'avg_rouge_l', 'avg_bert_f1'],
                         var_name='Metric', value_name='Score')
    
    # Đổi tên metric cho dễ đọc
    metric_map = {
        'avg_bleu': 'BLEU',
        'avg_rouge_l': 'ROUGE-L',
        'avg_bert_f1': 'BERTScore'
    }
    metrics_df['Metric'] = metrics_df['Metric'].map(metric_map)

    # 3. Vẽ biểu đồ so sánh Metrics
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=metrics_df, x='Metric', y='Score', hue='mode_name', palette='viridis')
    
    plt.title('So sánh chất lượng sinh văn bản (NLP Metrics)', fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, 1.1)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.xlabel('Chỉ số đánh giá', fontsize=12)
    plt.legend(title='Cấu hình mô hình', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Thêm giá trị số trên đầu cột
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "nlp_metrics_comparison.png", dpi=300)
    print(f"Đã lưu biểu đồ Metrics tại: {output_dir / 'nlp_metrics_comparison.png'}")

    # 4. Vẽ biểu đồ Recall@5 (Chỉ so sánh các chế độ có RAG)
    rag_only_df = df[df['avg_recall_at_5'].notnull()]
    if not rag_only_df.empty:
        plt.figure(figsize=(8, 6))
        ax2 = sns.barplot(data=rag_only_df, x='mode_name', y='avg_recall_at_5', palette='magma')
        plt.title('Hiệu năng truy xuất (Recall@5)', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.ylabel('Recall@5 Score')
        
        for p in ax2.patches:
            ax2.annotate(format(p.get_height(), '.3f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
        
        plt.tight_layout()
        plt.savefig(output_dir / "retrieval_recall_comparison.png", dpi=300)
        print(f"Đã lưu biểu đồ Recall tại: {output_dir / 'retrieval_recall_comparison.png'}")

    # 5. Vẽ biểu đồ Thời gian phản hồi
    plt.figure(figsize=(10, 6))
    ax3 = sns.barplot(data=df, x='mode_name', y='avg_generation_time', palette='coolwarm')
    plt.title('Thời gian phản hồi trung bình (Giây)', fontsize=14, fontweight='bold')
    plt.ylabel('Giây (Seconds)')
    
    for p in ax3.patches:
        ax3.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
                    
    plt.tight_layout()
    plt.savefig(output_dir / "response_time_comparison.png", dpi=300)
    print(f"Đã lưu biểu đồ thời gian tại: {output_dir / 'response_time_comparison.png'}")

if __name__ == "__main__":
    main()
