import json
from typing import Any
import string
from tqdm import tqdm

def normalize_str(s: str) -> str:
    """
    Normalize a string by removing punctuation and whitespace,
    and converting it to uppercase.
    """
    translator = str.maketrans('', '', string.punctuation + string.whitespace)
    return s.translate(translator).upper()

def normalize_json(obj: Any) -> Any:
    """
    Recursively normalize a JSON-like object:
      - For dict: normalize keys (if string) and values (order of keys is ignored)
      - For list: normalize each element (order preserved)
      - For str: normalize using normalize_str()
      - For other types: leave unchanged
    """
    if isinstance(obj, dict):
        normalized = {}
        for key, value in obj.items():
            new_key = normalize_str(key) if isinstance(key, str) else key
            normalized[new_key] = normalize_json(value)
        return normalized
    elif isinstance(obj, list):
        return [normalize_json(item) for item in obj]
    elif isinstance(obj, str):
        return normalize_str(obj)
    else:
        return obj

def wildcard_compare(val1: Any, val2: Any) -> bool:
    """
    Recursively compare two normalized JSON values.
    If either value is the wildcard string "$$$", consider it a match.
    """
    if isinstance(val1, str) and val1 == "$$$":
        return True
    if isinstance(val2, str) and val2 == "$$$":
        return True

    if isinstance(val1, dict) and isinstance(val2, dict):
        if set(val1.keys()) != set(val2.keys()):
            return False
        for k in val1.keys():
            if not wildcard_compare(val1[k], val2[k]):
                return False
        return True

    if isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return False
        unmatched = list(val2)
        for item1 in val1:
            found = False
            for i, item2 in enumerate(unmatched):
                if wildcard_compare(item1, item2):
                    found = True
                    del unmatched[i]
                    break
            if not found:
                return False
        return True

    return val1 == val2

def compare_json_with_wildcard(json1: Any, json2: Any) -> bool:
    """
    Compare two JSON objects by normalizing them and treating any value equal to "$$$" as a wildcard.
    """
    if json1 is None or json2 is None:
        return False
    try:
        json_obj1 = json.loads(json1)
        json_obj2 = json.loads(json2)
    except (TypeError, json.JSONDecodeError):
        # If json_obj1 is not a valid JSON string, return False
        return False
    norm1 = normalize_json(json_obj1)
    norm2 = normalize_json(json_obj2)
    return wildcard_compare(norm1, norm2)
