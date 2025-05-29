import json
from datasets import load_dataset

import re


def split_args(s):
    """
    Split a string like 'location="San Francisco", unit="Fahrenheit"'
    into ['location="San Francisco"', 'unit="Fahrenheit"'], 
    ignoring commas inside quotes.
    """
    args, current = [], ""
    in_quote, quote_char = False, ""
    for ch in s:
        if ch in ('"', "'"):
            if not in_quote:
                in_quote, quote_char = True, ch
            elif quote_char == ch:
                in_quote = False
        if ch == "," and not in_quote:
            args.append(current)
            current = ""
        else:
            current += ch
    if current:
        args.append(current)
    return [a.strip() for a in args if a.strip()]

def parse_output(output_str):
    """
    Parse '[func(a="x", b="y")]' → ('func', {'a':'x','b':'y'})
    Raises ValueError on totally wrong format.
    """
    pattern = r'^\s*\[?\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(\s*(.*)\)\s*\]?\s*$'
    # print(output_str)
    if output_str == None :
        raise ValueError("Invalid format")
    m = re.match(pattern, output_str)
    if not m:
        raise ValueError("Invalid format")
    func, args_str = m.groups()
    out_args = {}
    for arg in split_args(args_str):
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        v = v.strip()
        # strip matching quotes
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out_args[k.strip()] = v
    return func, out_args

def normalize(val):
    """Turn None → '' and everything else into its str()."""
    return "" if val is None else str(val)

def matches(value, expected_list):
    """
    True if value matches any expected (case-insensitive eq or substring).
    Both sides normalized via `normalize()`.
    """
    v = normalize(value).strip().lower()
    for exp in expected_list:
        e = normalize(exp).strip().lower()
        if v == e or v in e or e in v:
            return True
    return False

def is_optional_list(expected_list):
    """
    True if every expected value is only '' or None.
    """
    normalized = { normalize(exp).strip() for exp in expected_list }
    return normalized.issubset({""})

def evaluate(output_str, ground_truth_str):
    """
    Returns (True, "Match") or (False, error_msg).
    Never raises on bad format—just returns False.
    """
    try:
        func, out_args = parse_output(output_str)
    except ValueError:
        return False, "Invalid output format"

    # load GT
    gts = json.loads(ground_truth_str)
    gt_entry = next((d for d in gts if func in d), None)
    if not gt_entry:
        return [False, f"No ground truth for function '{func}'"]

    exp_args = gt_entry[func]
    errors = []

    for key, exp_vals in exp_args.items():
        if is_optional_list(exp_vals):
            # allowed to omit
            continue
        if key not in out_args:
            errors.append(f"Missing required parameter '{key}'")
        elif not matches(out_args[key], exp_vals):
            errors.append(
                f"Parameter '{key}' value '{out_args[key]}' not in {exp_vals}"
            )

    if errors:
        return [False, "; ".join(errors)]
    return [True, "Match"]





def load_bfcl_v2_ast_dataset(split: str = "train", max_samples: int = None):
    """
    Load and process the bfcl_v2_ast dataset.

    Args:
        split (str): Which split to load ("train", "test", "validation", etc.).
        max_samples (int, optional): If provided, only load up to this many samples.

    Returns:
        List[Dict]: A list where each item is a dict with:
                    - "input":      the source code or text input
                    - "ast":        the corresponding abstract syntax tree
    """
    # 1. Download & load the specified split from the Hugging Face Hub
    #    (automatically cached locally) :contentReference[oaicite:0]{index=0}
    dataset = load_dataset("hjshah/bfcl_v2_ast", split=split)

    # 2. Optionally truncate to the first `max_samples` entries
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    processed = []
    for entry in dataset:
        # Assuming the dataset has columns "input" and "ast"
        # Adjust these keys if the dataset uses different column names :contentReference[oaicite:1]{index=1}
        input_data = entry.get("sys_prompt")
        question_data = entry.get("question")
        ground_truth = entry.get("ground_truth")

        processed.append({
            "prompt": input_data,
            "question": question_data,
            "ground_truth": ground_truth
        })

    return processed

def test_bfcl_v2_ast(generation, ground_truth):
    """
    Test the generated output against the ground truth.

    Args:
        generation (str): The generated output.
        ground_truth (str): The expected output.

    Returns:
        bool: True if the generation matches the ground truth, False otherwise.
    """

    return evaluate(generation, ground_truth)

    

# if __name__ == "__main__":
#     # Example usage
#     data = load_bfcl_v2_ast_dataset(split="train")
#     print(f"Loaded {len(data)} samples from bfcl_v2_ast test split.")
#     print("First sample:", data[0])


# if __name__ == "__main__":
#     output = "[{'name': 'todo'}, todo(type='delete', content='ravi')]"
#     ground_truth = '''[{"todo": {"type": ["add"], "content": ["go to Goa"]}}]'''
#     ok, msg = evaluate(output, ground_truth)
#     print(ok, msg)  # → True, "Match"