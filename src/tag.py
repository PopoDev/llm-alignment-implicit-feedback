import os
import csv
import json
import argparse
import spacy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CURRENT_DIR, "../dataset")

nlp = spacy.load("en_core_web_sm")


def truncate(text, limit=256, start=True):
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    if start:
        cut = text[:limit]
        last_space = cut.rfind(" ")
        return cut[:last_space].strip() if last_space != -1 else cut.strip()
    else:
        cut = text[-limit:]
        first_space = cut.find(" ")
        return cut[first_space + 1 :].strip() if first_space != -1 else cut.strip()


def process_user_content(content):
    doc = nlp(content)
    verb = next((t.lemma_ for t in doc if t.pos_ in ("AUX", "VERB")), "")
    noun_chunks = set()

    for chunk in doc.noun_chunks:
        words = [
            token.text
            for token in chunk
            if not token.is_stop
            and token.is_alpha
            and token.pos_ not in {"DET", "PRON", "VERB", "AUX"}
        ]
        cleaned_text = " ".join(list(filter(lambda w: len(w) > 2, words)))
        if not cleaned_text:
            continue
        noun_chunks.add(cleaned_text)

    nouns = list(noun_chunks)[:10]

    return {
        "role": "user",
        "verb": verb,
        "nouns": nouns,
        "content": content,
    }


def process_conversation(conversation_json):
    try:
        conversation = json.loads(conversation_json)
    except json.JSONDecodeError:
        return "[]"
    for i, msg in enumerate(conversation):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            conversation[i] = process_user_content(content)

    conversation_lines = [json.dumps(s, ensure_ascii=False) for s in conversation]
    return "[\n" + ",\n".join(conversation_lines) + "\n]"


def save_outputs(rows, fieldnames, file):
    csv_file = f"{file}.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset name")
    args = parser.parse_args()
    dataset = args.dataset

    input_file = os.path.join(INPUT_DIR, f"{dataset}.csv")
    output_file = os.path.join(INPUT_DIR, f"{dataset}_tag")
    rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["conversation"] = process_conversation(row["conversation"])
            rows.append(row)
    fieldnames = list(rows[0].keys())
    save_outputs(rows, fieldnames, output_file)
    print(f"Summary of {dataset} with {len(rows)} rows saved to {output_file}")


if __name__ == "__main__":
    main()
