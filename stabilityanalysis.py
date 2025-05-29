import os
import sys
import json
from contextlib import ExitStack
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


path="stab/"
# model= "llama-3.1-8b-1-" 
model= "qwen-2.5-"
decoder="none"  # Assuming no specific decoder is used
# decoder="outlines"  
# decoder="llguidance"
# Filenames for the five JSONL files (0 through 4)
file_names = [path+model+decoder+f"-{i}-spider.jsonl" for i in range(5)]

def compute_bleu(reference, candidate):
    reference_tokens = reference.strip().split()
    candidate_tokens = candidate.strip().split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

try:
    with ExitStack() as stack:
        files = [stack.enter_context(open(fname, 'r')) for fname in file_names]

        line_counts = []
        for f in files:
            count = sum(1 for _ in f)
            line_counts.append(count)
            f.seek(0)
        if len(set(line_counts)) != 1:
            print(f"Warning: File line counts are not equal: {line_counts}")
            print("Proceeding with evaluation up to the minimum common length.")
        total_tasks = min(line_counts)

        pass_count = 0
        pass1_counts = [0] * 5  # Individual pass@1 counters for each file
        total_counted = 0
        bleu_sum = 0.0
        bleu_count = 0

        for lines in zip(*files):
            if total_counted >= total_tasks:
                break

            try:
                objs = [json.loads(line) for line in lines]
            except json.JSONDecodeError as e:
                print(f"JSON parse error at task {total_counted}: {e}")
                total_counted += 1
                continue

            match_flags = []
            for obj in objs:
                if not isinstance(obj, dict) or "match" not in obj:
                    match_flags.append(False)
                else:
                    flag = False
                    if isinstance(obj["match"], dict) and "match" in obj["match"]:
                        flag = bool(obj["match"]["match"])
                    match_flags.append(flag)

            if any(match_flags):
                pass_count += 1

            for i in range(5):
                if match_flags[i]:
                    pass1_counts[i] += 1

            if not isinstance(objs[0], dict) or "result" not in objs[0]:
                total_counted += 1
                continue
            ref_query = str(objs[0]["result"]).strip()

            for j in range(1, len(objs)):
                if not isinstance(objs[j], dict) or "result" not in objs[j]:
                    continue
                pred_query = str(objs[j]["result"]).strip()
                try:
                    bleu_score = compute_bleu(ref_query, pred_query)
                except Exception as e:
                    print(f"Warning: BLEU computation failed at task {total_counted} for file index {j}: {e}")
                    continue
                bleu_sum += bleu_score
                bleu_count += 1

            total_counted += 1

        pass_rate = pass_count / total_tasks if total_tasks > 0 else 0.0
        avg_bleu = bleu_sum / bleu_count if bleu_count > 0 else 0.0
        avg_bleu_dist = 1.0 - avg_bleu

        print(f"pass@5: {pass_count}/{total_tasks} = {pass_rate:.2%}")
        for i, count in enumerate(pass1_counts):
            rate = count / total_tasks if total_tasks > 0 else 0.0
            print(f"pass@1 (file {i}): {count}/{total_tasks} = {rate:.2%}")
        print(f"Average BLEU similarity (files 1-4 vs file 0): {avg_bleu:.4f}")
        print(f"Average BLEU distance (1 - similarity): {avg_bleu_dist:.4f}")
except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")