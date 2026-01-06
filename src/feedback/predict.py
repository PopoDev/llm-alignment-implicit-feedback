import csv
import json
import argparse
import sys
from dataclasses import dataclass, asdict, fields
from pathlib import Path

csv.field_size_limit(sys.maxsize)
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm import call_llm


def predict_feedback(conversation_json, prompt):
    """Process conversation turn-by-turn, return dialog acts and feedback array."""
    try:
        conversation = json.loads(conversation_json)
    except json.JSONDecodeError:
        return [], []

    dialog_acts = ["question"]  # Initial user message is always a question
    feedbacks = []
    current_feedback = ""

    initial_msg = conversation[0]
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(initial_msg)},
    ]

    # Process pairs: assistant + user
    i = 1
    while i < len(conversation) - 1:
        assistant_msg = conversation[i]
        user_msg = conversation[i + 1]

        turn_input = [assistant_msg, user_msg]
        # print(turn_input)

        # Add to messages
        messages.append({"role": "user", "content": json.dumps(turn_input)})
        # print(messages[1:])

        # Call API
        response_json = call_llm(messages)
        messages.append({"role": "assistant", "content": json.dumps(response_json)})
        print(response_json)

        # Check if topic change
        if "open_feedback" in response_json:
            current_feedback = response_json["open_feedback"]
            feedbacks.append(current_feedback)

            # Reset messages for new conversation
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(user_msg)},
            ]

            # New conversation starts with question
            if i < len(conversation) - 1:
                dialog_acts.extend(["SWITCH", "question"])
        else:
            dialog_acts.append(response_json["dialog_act"])

        i += 2

    return dialog_acts, feedbacks


@dataclass
class ConversationRow:
    conversation_id: str
    turn: str
    predicted_dialog: str
    predicted_switch: str
    conversation: str
    metadata: str = ""

    @classmethod
    def fieldnames(cls):
        return [f.name for f in fields(cls)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", default="data/dataset.csv", help="CSV file to process"
    )
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    input_file = args.filename
    base_name = Path(input_file).stem
    output_file = Path(f"data/feedback/{base_name}.csv")
    prompt_file = current_dir / "prompt.txt"

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    input_rows = []
    with open(input_file, newline="", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))

    start_idx = 0
    exists = output_file.exists()
    if exists:
        with open(output_file, newline="", encoding="utf-8") as f:
            # Check if file is empty or just header
            try:
                reader = csv.DictReader(f)
                existing_rows = sum(1 for _ in reader)
                start_idx = max(existing_rows, 0)
            except Exception as e:
                print(f"Error reading output file to resume: {e}")
                start_idx = 0
        print(f"Resuming from row {start_idx + 1} of {len(input_rows)} total.")

    write_header = not exists
    with open(output_file, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=ConversationRow.fieldnames())
        if write_header:
            writer.writeheader()

        for idx, row in enumerate(input_rows):
            if idx < start_idx:
                continue

            print(f"\n[{idx}/{len(input_rows)}] Processing: {row['conversation_id']}")
            try:
                dialog_acts, feedbacks = predict_feedback(row["conversation"], prompt)

                conversation_row = ConversationRow(
                    conversation_id=row["conversation_id"],
                    turn=row["turn"],
                    predicted_dialog=json.dumps(dialog_acts),
                    predicted_switch=json.dumps(feedbacks),
                    conversation=row["conversation"],
                    metadata=row.get("metadata", ""),
                )

                writer.writerow(asdict(conversation_row))
                f_out.flush()
                print(f"Dialog acts: {dialog_acts}")
                print(f"Feedbacks: {feedbacks}")
            except Exception as e:
                print(f"Error processing row {idx+1}: {e}")
                continue

    print(f"\nFeedback prediction saved to: {output_file}")


if __name__ == "__main__":
    main()
