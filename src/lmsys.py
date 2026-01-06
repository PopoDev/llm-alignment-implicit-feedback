import os
import json
import csv
from datasets import load_dataset

COLUMNS = ["conversation_id", "turn", "conversation"]


def normalize_row(row):
    conversation_lines = [
        json.dumps({"role": msg["role"], "content": msg["content"]}, ensure_ascii=False)
        for msg in row["conversation"]
    ]

    return {
        "conversation_id": row["conversation_id"],
        "turn": row["turn"],
        "conversation": "[\n" + ",\n".join(conversation_lines) + "\n]",
    }


def save_to_csv(dataset, dir, filename):
    path = os.path.join(dir, f"{filename}.csv")
    csv.field_size_limit(131072)  # Ensure we are using the default limit for checking

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in dataset:
            normalized = normalize_row(row)
            if len(normalized["conversation"]) > 130000: # Leave some buffer
                continue
            writer.writerow(normalized)


def main():
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train")

    english_ds = ds.filter(lambda x: x["language"] == "English")
    print(f"English conversations: {(len(english_ds) * 100 / len(ds)):.2f}%")

    clean_ds = english_ds.filter(
        lambda x: (x["turn"] > 1)
        and all(not m.get("flagged", False) for m in x["openai_moderation"])
    )
    print(f"Clean conversations: {(len(clean_ds) * 100 / len(english_ds)):.2f}%")

    processed_ds = clean_ds.select(range(1000))
    processed_ds = processed_ds.remove_columns(
        [col for col in processed_ds.column_names if col not in COLUMNS]
    )

    dir = "dataset"
    filename = "lmsys"

    os.makedirs(dir, exist_ok=True)

    save_to_csv(processed_ds, dir, filename)

    print(f"Saved 1000 samples to {dir}/{filename}")


if __name__ == "__main__":
    main()
