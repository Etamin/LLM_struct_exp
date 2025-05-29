from datasets import load_dataset

def load_humaneval(split="test"):
    """
    Load the HumanEval dataset split from Hugging Face.

    Returns:
        List[dict]: Each dict contains fields:
            - task_id (str)
            - prompt (str)
            - canonical_solution (str)
            - test (str)
            - entry_point (str)
    """
    # The HumanEval dataset is available via Hugging Face Datasets
    ds = load_dataset("openai/openai_humaneval", split=split)  # citeturn0search1
    return [dict(example) for example in ds]

def test_humaneval_problem(problem: dict, completion: str) -> bool:
    """
    Given a HumanEval problem dict and a generated completion string,
    compile and run the test code. Returns True if the test passes, False otherwise.
    """
    # Construct full source: prompt (function header) + completion (function body)
    source_code = problem["prompt"] + completion.replace("```python", "").replace("```", "")
    local_vars = {}
    try:
        # Define the candidate function
        exec(source_code, {}, local_vars)
        # Execute the provided test harness
        exec(problem["test"], {}, local_vars)  # citeturn0search5
        # Call the check function with the entry point
        entry_fn = local_vars.get(problem["entry_point"])
        local_vars["check"](entry_fn)
        return True
    except Exception:
        return False

# if __name__ == '__main__':
#     # Example usage:
#     problems = load_humaneval()
#     first_problem = problems[112]
#     result = test_humaneval_problem(first_problem, first_problem["canonical_solution"])
#     print("Passes test:", result)
