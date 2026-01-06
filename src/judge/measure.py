import json
import os
import sys
from pathlib import Path
import statistics

def measure_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    total = len(data)
    if total == 0:
        print(f"No data in {file_path.name}")
        return

    better_count = 0
    scores = []

    for item in data:
        improvement = item.get("improvement", "").lower()
        # Check for "better"
        if improvement.lower() == "better":
            better_count += 1
        
        score_str = item.get("score")
        if score_str:
            try:
                scores.append(float(score_str))
            except ValueError:
                pass
    
    win_rate = (better_count / total) * 100 if total > 0 else 0
    mean_score = statistics.mean(scores) if scores else 0
    
    print(f"Results for {file_path.name}:")
    print(f"  Total samples: {total}")
    print(f"  Win rate: {win_rate:.2f}% ({better_count}/{total})")
    print(f"  Mean score: {mean_score:.2f}")
    print("-" * 30)

    # Save results to file
    results_dir = file_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = results_dir / f"{file_path.stem}_results.json"
    
    result_data = {
        "filename": file_path.name,
        "total_samples": total,
        "win_rate": win_rate,
        "mean_score": mean_score
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
        print(f"Saved results to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")

def main():
    # If arguments provided, process those files
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            path = Path(arg)
            if path.exists():
                measure_file(path)
            else:
                print(f"File not found: {path}")
    else:
        base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "judge"
        
        if not base_dir.exists():
            print(f"Directory not found: {base_dir}")
            return
            
        found_files = False
        for filename in sorted(os.listdir(base_dir)):
            if filename.endswith(".json"):
                measure_file(base_dir / filename)
                found_files = True
        
        if not found_files:
            print(f"No JSON files found in {base_dir}")

if __name__ == "__main__":
    main()