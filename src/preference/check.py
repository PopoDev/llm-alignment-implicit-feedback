import os
import csv
import json
from pathlib import Path

DIALOG_ACT = [
    "SWITCH",
    "inform",
    "correction",
    "confirm",
    "question",
    "request",
    "greeting",
    "none",
]

COLUMNS = [
    "conversation_id",
    "turn",
    "predicted_dialog_act",
    "predicted_feedback",
    "conversation",
]


def is_empty(dialog):
    """Return True if dialog_act is empty"""
    if not dialog:
        return True
    stripped = dialog.strip()
    return stripped == "[]" or stripped == ""


def check_csv(input_path):
    cleaned_rows = []
    columns = None

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames

        if columns != COLUMNS:
            print(f"⚠️ Warning: Unexpected columns in {input_path}: {columns}")

        for idx, row in enumerate(reader, start=2):
            dialog_value = row.get("dialog_act", "")

            if is_empty(dialog_value):
                continue

            try:
                acts = json.loads(dialog_value)
            except json.JSONDecodeError:
                print(f"⚠️ Skip invalid JSON in row {idx}: {dialog_value}")
                continue

            invalid_acts = [a for a in acts if a not in DIALOG_ACT]
            if invalid_acts:
                print(f"⚠️ Skip invalid acts in row {idx}: {invalid_acts}")
                continue

            num_turns = int(row.get("turn", 0))
            if len(acts) < num_turns:
                print(
                    f"⚠️ Skip invalid acts in row {idx}: {invalid_acts} expected at least {num_turns}"
                )
                continue

            row["dialog_act"] = json.dumps(acts)
            cleaned_rows.append(row)

    return cleaned_rows, columns


def main():
    input_dir = Path(__file__).parent.parent.parent / "data" / "feedback"

    for filename in os.listdir(input_dir):
        if not filename.endswith(".csv"):
            continue

        input_path = input_dir / filename

        print(f"Checking {filename}...")
        check_csv(input_path)


if __name__ == "__main__":
    main()
