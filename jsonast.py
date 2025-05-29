import json
import re
from typing import Any, Tuple, Union, List, Optional


def _normalize_whitespace(obj: Any) -> Any:
    """
    Recursively normalize whitespace in all string values within the JSON-like object:
      - Collapse all whitespace sequences (including Unicode spaces) to a single ASCII space
      - Strip leading/trailing whitespace
    """
    if isinstance(obj, dict):
        return {k: _normalize_whitespace(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_whitespace(v) for v in obj]
    elif isinstance(obj, str):
        # Collapse any sequence of whitespace characters to a single space
        normalized = re.sub(r"\s+", " ", obj, flags=re.UNICODE).strip()
        return normalized
    else:
        return obj

def remove_quotes_from_ends(s: str) -> str:
    """
    Removes single or double quotes from the beginning and end of the string.
    If there is just one quote (single or double) at the start or end, it will also be removed.
    
    Args:
        s: The string from which quotes are to be removed.
        
    Returns:
        The string with leading/trailing quotes removed.
    """
    # Remove leading and trailing quotes if they exist (either single or double quotes)
    if s.startswith(("'", '"')) and s.endswith(("'", '"')):
        return s[1:-1]
    # Handle cases where only one quote is at the start or end
    if s.startswith("'") or s.startswith('"'):
        return s[1:]
    if s.endswith("'") or s.endswith('"'):
        return s[:-1]
    return s
    
def compare_jsons(
    json_text1: str,
    json_text2: str,
    ignore_values: Optional[Union[Any, List[Any]]] = None
) -> Tuple[bool, bool, bool]:
    """
    Compare two JSON texts for:
      1) syntax validity,
      2) structure equality (ignoring values),
      3) value equality with optional ignore list.

    Args:
        json_text1: First JSON string.
        json_text2: Second JSON string.
        ignore_values: A value or list of values to ignore when comparing;
                       if a leaf value in either JSON matches any in this list,
                       it's considered equal.

    Returns:
        (syntax_ok, structure_ok, values_match)
    """
    # 1) Syntax check
    if json_text1 is None or json_text2 is None:
        return False, False, False
    try:
        # Normalize single quotes and extra chars
        json_text1 = json_text1.replace("'", '"').replace("\"", '"').replace("\n", " ").replace("```json", "").replace("```", "")
        json_text2 = json_text2.replace("'", '"').replace("\"", '"').replace("\\", '').replace("\n", " ").replace("```json", "").replace("```", "")
        # Load the JSON objects

        obj1 = json.loads(remove_quotes_from_ends(json_text1))
        obj2 = json.loads(remove_quotes_from_ends(json_text2))
    except json.JSONDecodeError:
        return False, False, False
    syntax_ok = True

    # Normalize whitespace in all string values
    obj1 = _normalize_whitespace(obj1)
    obj2 = _normalize_whitespace(obj2)

    # Helper: structure check
    def _same_structure(a: Any, b: Any) -> bool:
        if isinstance(a, dict) and isinstance(b, dict):
            # Sort keys to handle unordered nature of JSON objects
            if set(a.keys()) != set(b.keys()):
                return False
            return all(_same_structure(a[k], b[k]) for k in a.keys())
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            return all(_same_structure(x, y) for x, y in zip(a, b))
        else:
            return not (isinstance(a, (dict, list)) or isinstance(b, (dict, list)))

    structure_ok = _same_structure(obj1, obj2)

    # Prepare ignore list
    if ignore_values is None:
        ignore_list: Optional[List[Any]] = None
    else:
        ignore_list = list(ignore_values) if isinstance(ignore_values, list) else [ignore_values]

    # Helper: deep equality with ignore
    def _deep_equal(a: Any, b: Any) -> bool:
        # If either value is in ignore_list, treat as equal
        if ignore_list is not None and (a in ignore_list or b in ignore_list):
            return True
        # Type mismatch -> not equal
        if type(a) != type(b):
            return False
        if isinstance(a, dict):
            for key in a:
                if key not in b or not _deep_equal(a[key], b[key]):
                    return False
            return True
        if isinstance(a, list):
            if len(a) != len(b):
                return False
            return all(_deep_equal(x, y) for x, y in zip(a, b))
        # Leaf: direct comparison
        return a == b

    values_match = _deep_equal(obj1, obj2)

    return syntax_ok, structure_ok, values_match


# if __name__ == "__main__":
    # Examples
    # a="{\"API\": [\"getAccountID\"], \"parameters\": [{\"accountNumber\": \"123456\"}]}"
    # b=" {'API': ['getAccountID'], 'parameters': [{'accountNumber': '123456'}]}"
    # Ignore whitespace only
    # print(compare_jsons(a, b))
    # Ignore specific value
    # print(compare_jsons(a, c, ignore_values="bar"))  # True on value match
    # # Ignore list of values
    # print(compare_jsons(a, c, ignore_values=["bar", 42]))
    # print(compare_jsons(a, '{"x": [1, {"z": 3}], "z": "foo"}'))  # False structure
