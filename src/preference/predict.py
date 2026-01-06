import csv
import json
import argparse
import sys
from dataclasses import dataclass, asdict, fields
from pathlib import Path

csv.field_size_limit(sys.maxsize)
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm import call_llm


def predict_preference(conversation_json, dialog_acts, prompt):
    """Process conversation turn-by-turn, return the improved response for the preference data."""
    try:
        conversation = json.loads(conversation_json)
        if isinstance(dialog_acts, str):
            dialog_acts = json.loads(dialog_acts)
    except json.JSONDecodeError:
        return []

    predicted_preference = []

    initial_msg = conversation[0]
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(initial_msg)},
    ]

    # Process pairs: assistant + user
    turn = 1
    i = 1
    while i < len(conversation) - 1:
        assistant_msg = conversation[i]
        user_msg = conversation[i + 1]

        dialog_act = dialog_acts[turn]
        turn_input = [assistant_msg, user_msg, dialog_act]
        # print(turn_input)

        # Add to messages
        messages.append({"role": "user", "content": json.dumps(turn_input)})
        # print(messages[1:])

        # Check if topic change
        if dialog_act == "SWITCH":
            # Reset messages for new conversation
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(user_msg)},
            ]
            turn += 1
            continue

        # Call API 
        if dialog_act in ["correction", "request"]:
            print(turn, dialog_act)
            response_json = call_llm(messages)
            # messages.append({"role": "assistant", "content": json.dumps(response_json)})
            print(response_json)
            
            improved_response = response_json.get("improved_response", "")
            predicted_preference.append(improved_response)
        else:
            predicted_preference.append("") 
            
        
        turn += 1
        i += 2
    
    return predicted_preference


@dataclass
class ConversationRow:
    conversation_id: str
    turn: str
    predicted_dialog: str
    predicted_switch: str
    predicted_preference: str
    conversation: str
    metadata: str = ""

    @classmethod
    def fieldnames(cls):
        return [f.name for f in fields(cls)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", default="data/feedback/dataset.csv", help="CSV file to process"
    )
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    input_file = args.filename
    base_name = Path(input_file).stem
    output_file = Path(f"data/preference/{base_name}.csv")
    prompt_file = current_dir / "prompt.txt"

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    start_idx = 0
    exists = output_file.exists()
    if exists:
        with open(output_file, newline="", encoding="utf-8") as f:
            existing_reader = list(csv.DictReader(f))
            existing_rows = len(existing_reader)
            start_idx = max(existing_rows, 0)
        print(f"Resuming from row {start_idx + 1}.")

    write_header = not exists
    with open(output_file, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=ConversationRow.fieldnames())
        if write_header:
            writer.writeheader()

        for idx, row in enumerate(reader):
            if idx < start_idx:
                continue

            print(f"\n[{idx}/{len(reader)}] Processing: {row['conversation_id']}")
            try:
                conversation = row["conversation"]
                predicted_preference = predict_preference(conversation, row["predicted_dialog"], prompt)

                conversation_row = ConversationRow(
                    conversation_id=row["conversation_id"],
                    turn=row["turn"],
                    predicted_dialog=row["predicted_dialog"],
                    predicted_switch=row["predicted_switch"],
                    predicted_preference=json.dumps(predicted_preference),
                    conversation=conversation,
                    metadata=row.get("metadata", ""),
                )

                writer.writerow(asdict(conversation_row))
                f_out.flush()

                print(f"Predicted preference: {json.dumps(predicted_preference, indent=2)}")
            except Exception as e:
                print(f"Error processing row {idx+1}: {e}")
                continue

    print(f"\nPreference prediction saved to: {output_file}")


if __name__ == "__main__":
    main()
