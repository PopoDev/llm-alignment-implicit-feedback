import sys
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm import call_llm


def predict_judge(conversation_pair, prompt_template):
    conversation_history = conversation_pair["prompt"]["content"]
    original_response = conversation_pair["rejected"]["content"]
    improved_response = conversation_pair["chosen"]["content"]

    prompt = prompt_template.replace("{{prompt}}", conversation_history)
    prompt = prompt.replace("{{rejected}}", original_response)
    prompt = prompt.replace("{{chosen}}", improved_response)

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages)
    print(response)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", default="data/dpo/dataset.jsonl", help="JSON file to process"
    )
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    input_file = Path(args.filename)
    base_name = input_file.stem
    output_file = Path(f"data/judge/{base_name}.json")
    prompt_file = current_dir / "prompt.txt"

    with open(prompt_file, "r") as f:
        prompt_template = f.read()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as out_f:
        out_f.write("[\n")
        first = True
        
        with open(input_file, "r") as f:
            for line in f:
                try:
                    conversation_pair = json.loads(line)
                    evaluation = predict_judge(conversation_pair, prompt_template)
                    if evaluation:
                        conversation_pair.update(evaluation)
                        
                        if not first:
                            out_f.write(",\n")
                        else:
                            first = False
                            
                        json.dump(conversation_pair, out_f, indent=2)
                        out_f.flush()
                        
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
                except Exception as e:
                    print(f"Error processing line: {e}")
        
        out_f.write("\n]")
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
