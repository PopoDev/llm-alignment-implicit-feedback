import os
import json
import csv
import argparse
from datasets import load_dataset

COLUMNS = ["conversation_hash", "timestamp", "hashed_ip", "country", "turn", "conversation"]


def normalize_row(row):
    conversation_lines = [
        json.dumps({"role": msg["role"], "content": msg["content"]}, ensure_ascii=False)
        for msg in row["conversation"]
    ]

    metadata = json.dumps({
        "hashed_ip": row["hashed_ip"],
        "country": row["country"],
        "timestamp": str(row["timestamp"])
    }, ensure_ascii=False)

    return {
        "conversation_id": row["conversation_hash"],
        "turn": row["turn"],
        "conversation": "[\n" + ",\n".join(conversation_lines) + "\n]",
        "metadata": metadata,
    }


def save_to_csv(dataset, dir, filename):
    path = os.path.join(dir, f"{filename}.csv")
    csv.field_size_limit(131072)  # Ensure we are using the default limit for checking

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["conversation_id", "turn", "conversation", "metadata"])
        writer.writeheader()
        for row in dataset:
            normalized = normalize_row(row)
            if len(normalized["conversation"]) > 130000: # Leave some buffer
                continue
            writer.writerow(normalized)


def main():
    parser = argparse.ArgumentParser(description="Preprocess WildChat-1M dataset")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to extract (default: 100)"
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="dsc",
        choices=["asc", "dsc"],
        help="Sort by timestamp: 'asc' for ascending, 'dsc' for descending (default: dsc)"
    )
    args = parser.parse_args()

    ds = load_dataset("allenai/WildChat-1M", split="train")

    english_ds = ds.filter(lambda x: x["language"] == "English")
    print(f"English conversations: {(len(english_ds) * 100 / len(ds)):.2f}%")

    clean_ds = english_ds.filter(
        lambda x: (x["turn"] > 1)
        and (not x["toxic"])
        and all(not m.get("flagged", False) for m in x["openai_moderation"])
    )
    print(f"Clean conversations: {(len(clean_ds) * 100 / len(english_ds)):.2f}%")

    # Sort by timestamp
    reverse = (args.sort == "dsc")
    sorted_ds = clean_ds.sort("timestamp", reverse=reverse)

    num_samples = min(args.num_samples, len(sorted_ds))
    processed_ds = sorted_ds.select(range(num_samples))
    processed_ds = processed_ds.remove_columns(
        [col for col in processed_ds.column_names if col not in COLUMNS]
    )

    dir = "dataset"
    filename = f"wildchat_{num_samples}"

    os.makedirs(dir, exist_ok=True)

    save_to_csv(processed_ds, dir, filename)

    print(f"Saved {num_samples} samples to {dir}/{filename}")


if __name__ == "__main__":
    main()
