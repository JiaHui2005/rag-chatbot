import json
import re
from collections import Counter
from pathlib import Path

FILE_PATH = Path("data/cleaned/phi3_finetune.jsonl")
FIXED_PATH = Path("data/cleaned/phi3_finetune_fixed.jsonl")

def fix_line(line):
    line = line.strip()
    if not line:
        return None
    
    # Try parsing as is
    try:
        return json.loads(line)
    except Exception:
        pass

    # Try to recover common patterns
    # Pattern 1: Missing ']' before category
    # {"messages": [{"role": "user", ...}, {"role": "assistant", ...}}, "category": "..."]
    if '}, "category":' in line and ']}' not in line:
        fixed = line.replace('}, "category":', '}], "category":')
        try:
            return json.loads(fixed)
        except Exception:
            pass

    # Pattern 2: Weird structure like line 29
    # {"role": "user", "content": "..."}, "messages": [{"role": "assistant", "content": "..."}], "category": "..."}
    # We want: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "category": "..."}
    match = re.match(r'({"role": "user", "content": ".*?"}), "messages": \[({"role": "assistant", "content": ".*?"})\], "category": "(.*?)"}', line)
    if match:
        user_obj = match.group(1)
        assistant_obj = match.group(2)
        category = match.group(3)
        fixed = f'{{"messages": [{user_obj}, {assistant_obj}], "category": "{category}"}}'
        try:
            return json.loads(fixed)
        except Exception:
            pass

    return None

def main():
    if not FILE_PATH.exists():
        print("File not found.")
        return

    fixed_records = []
    broken_count = 0
    recovered_count = 0
    
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Try to load
            try:
                data = json.loads(line)
                fixed_records.append(data)
            except:
                fixed_data = fix_line(line)
                if fixed_data:
                    fixed_records.append(fixed_data)
                    recovered_count += 1
                else:
                    broken_count += 1

    # Save fixed file
    with open(FIXED_PATH, 'w', encoding='utf-8') as f:
        for record in fixed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Processed {len(fixed_records) + broken_count} lines.")
    print(f"Successfully loaded: {len(fixed_records) - recovered_count}")
    print(f"Recovered: {recovered_count}")
    print(f"Discarded (too broken): {broken_count}")
    print(f"Total entries in fixed file: {len(fixed_records)}")

    # Check distribution
    categories = [r["category"] for r in fixed_records]
    counts = Counter(categories)
    total = len(categories)
    
    print("\nFixed Category Distribution:")
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

if __name__ == "__main__":
    main()
