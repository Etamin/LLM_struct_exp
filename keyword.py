import os
import json
import csv
from collections import Counter
from openai import OpenAI
from tqdm import tqdm
# ─── Configuration ─────────────────────────────────────────────────────────────
JSONL_FOLDER = "common_results"
OUTPUT_CSV    = "error_keyword_counts.csv"
os.environ["OPENAI_API_KEY"]="sk"

# CSV format definition:
#   Column 1: keyword (string)
#   Column 2: count   (integer)

# ─── Initialize OpenAI client ─────────────────────────────────────────────────
client = OpenAI()

def extract_keywords_from_error(error_message):
    """
    Send the error message to the LLM and return a list of extracted keywords.
    Raises RuntimeError on API failure.
    """
    prompt = (
        "Extract a concise, comma-separated list of keywords from the following "
        "error message. Focus on error types, function names, and data types. "
        "Only include terms present in the message.\n\n"
        f"```{error_message}```\n\nKeywords:"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60
        )
        content = resp.choices[0].message.content.strip()
        # Split on commas and strip whitespace
        # print(f"Extracted keywords: {content}")
        return [kw.strip() for kw in content.split(",") if kw.strip()]
    except Exception as e:
        raise RuntimeError(f"LLM keyword extraction failed: {e}")

def main():
    keyword_counts = Counter()

    # ─── Read and process each JSONL file ─────────────────────────────────────
    for fname in os.listdir(JSONL_FOLDER):
        if not fname.endswith(".jsonl") and ("bigcodebench"  in fname or "spider" in fname):
            continue
        path = os.path.join(JSONL_FOLDER, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in tqdm(enumerate(f, start=1)):
                    try:
                        record = json.loads(line)
                        match_obj = record.get("match", {})

                        # ── Skip success cases ─────────────────────────────────
                        # 1) category == "success"
                        # 2) match == True
                        if match_obj.get("category") == "success" or match_obj.get("match") is True:
                            continue

                        err = match_obj.get("error")
                        if not err:
                            continue

                        kws = extract_keywords_from_error(err)
                        keyword_counts.update(kws)
                    except json.JSONDecodeError as jde:
                        print(f"[WARN] Skipping malformed JSON in {fname}:{line_num}: {jde}")
                    except RuntimeError as rte:
                        print(f"[ERROR] {rte} -- skipping this error message.")
        except (IOError, OSError) as io_err:
            print(f"[ERROR] Cannot open file {path}: {io_err}")

    # ─── Write counts to CSV ───────────────────────────────────────────────────
        try:
            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["keyword", "count"])
                for kw, cnt in keyword_counts.most_common():
                    writer.writerow([kw, cnt])
            print(f"✔️  Keyword counts successfully written to {OUTPUT_CSV}")
        except (IOError, OSError) as io_err:
            print(f"[ERROR] Failed to write CSV file {OUTPUT_CSV}: {io_err}")

if __name__ == "__main__":
    main()