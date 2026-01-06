import os
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

def extract_pairs(csv_file):
    """
    Extract pairs for DPO training.
    Input: predicted_preference,conversation
    Output:
    {
        "prompt": "<conversation up to turn>", 
        "chosen": "<improved response>", 
        "rejected": "<original response>"
    }
    """
    pairs = []
    
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                predicted_preference = json.loads(row["predicted_preference"])
                conversation = json.loads(row["conversation"])
                num_turns = int(row["turn"])
                
                predicted_dialog = json.loads(row["predicted_dialog"])
                
                for i in range(num_turns-1):
                    # Filter: only take non-empty predicted_preference
                    if not predicted_preference[i]:
                        continue
                        
                    # Filter: only pairs where the initial act is "question"
                    if predicted_dialog[i] != "question":
                        continue

                    print(i)
                    print(predicted_dialog)
                    print(predicted_preference)

                    prompt = {"role": conversation[2*i]["role"], "content": conversation[2*i]["content"]}
                    chosen = {"role": "assistant", "content": predicted_preference[i]}
                    rejected = {"role": "assistant", "content": conversation[2*i+1]["content"]}
                    
                    pairs.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected
                    })
                    
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Error processing row: {e}")
                continue
                
    return pairs


def main():
    input_dir = Path(__file__).parent.parent.parent / "data" / "preference"
    output_dir = Path(__file__).parent.parent.parent / "data" / "dpo"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.endswith(".csv"):
            continue
            
        input_path = input_dir / filename
        dataset_name = filename.replace(".csv", "")
        output_path = output_dir / f"{dataset_name}.jsonl"
        
        print(f"Processing {filename}...")
        pairs = extract_pairs(input_path)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
                
        print(f"Saved {len(pairs)} pairs to {output_path}")


if __name__ == "__main__":
    main()