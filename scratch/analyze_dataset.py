import json
from collections import Counter
from pathlib import Path

FILE_PATH = Path("data/cleaned/phi3_finetune.jsonl")

def analyze():
    if not FILE_PATH.exists():
        print(f"File not found: {FILE_PATH}")
        return

    categories = []
    errors = []
    
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Check structure
                if "messages" not in data or "category" not in data:
                    errors.append(f"Line {i}: Missing 'messages' or 'category' keys.")
                elif not isinstance(data["messages"], list) or len(data["messages"]) != 2:
                    errors.append(f"Line {i}: 'messages' should be a list of 2 items.")
                else:
                    categories.append(data["category"])
            except Exception as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")

    total = len(categories)
    counts = Counter(categories)
    
    print(f"Total valid entries: {total}")
    print("\nCategory Distribution:")
    target_ratios = {
        "Direct QA": 0.40,
        "Paraphrase": 0.25,
        "Reasoning": 0.15,
        "Refusal": 0.10,
        "Out-of-scope": 0.10
    }
    
    for cat, ratio in target_ratios.items():
        count = counts.get(cat, 0)
        actual_ratio = count / total if total > 0 else 0
        print(f"  {cat:15}: {count:3} ({actual_ratio*100:4.1f}%) [Target: {ratio*100:4.1f}%]")

    if errors:
        print(f"\nErrors found ({len(errors)}):")
        for err in errors[:10]: # Show first 10
            print(f"  {err}")
    else:
        print("\nNo formatting errors found.")

if __name__ == "__main__":
    analyze()
