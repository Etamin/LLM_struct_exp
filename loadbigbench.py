from datasets import load_dataset  # citeturn0search0
import io
import unittest
import traceback
import multiprocessing
def load_bigcodebench(split="v0.1.4", prompt_type="instruct_prompt"):
    """
    Load the BigCodeBench dataset split and return a list of problem dicts.

    Each dict contains:
      - task_id (str)
      - prompt (str)
      - canonical_solution (str)
      - test_code (str)
    """
    ds = load_dataset("bigcode/bigcodebench", split=split)  # citeturn0search0
    problems = []
    for ex in ds:
        prompt = ex[prompt_type]
        solution = ex["canonical_solution"]
        test_code = ex["test"]
        problems.append({
            "task_id": ex["task_id"],
            "prompt": prompt,
            "canonical_solution": solution,
            "test_code": test_code,
            "code_prompt": ex["code_prompt"],
        })
    return problems

def test_bigcodebench_problem(problem, completion,timeout=10):
    """
    Given a BigCodeBench problem dict and a generated completion string,
    execute the prompt, the completion, and the test harness. 
    Returns True if all tests pass, False otherwise.
    """
    # Combine prompt (if it's code scaffold) and completion
    source = completion.replace("```python", "").replace("```", "")
    # Prepare a globals dict for execution
    globs = {}
    # print(completion)
    # Execute the source code
    try:
        exec(source, globs)
        # Execute the test harness
        exec(problem["test_code"], globs)  # citeturn0search6
        # Collect all TestCase subclasses
        test_cases = [
            obj for obj in globs.values()
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase)
        ]
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for tc in test_cases:
            suite.addTests(loader.loadTestsFromTestCase(tc))
        # Run the tests
        runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0)
        result = runner.run(suite)
        return result.wasSuccessful()
    except Exception:
        print("Error during test execution:", problem["task_id"])
        return False


def _worker(problem, completion, conn):
    """
    Worker function run in a separate process.
    Executes compile→exec→tests and sends back a dict result.
    """
    try:
        # 1) Strip fences and compile
        source = completion.replace("```python", "").replace("```", "")
        try:
            compiled = compile(source, '<bigcode>', 'exec')
        except SyntaxError as e:
            conn.send({
                "category": "syntax_error",
                "error": traceback.format_exception_only(type(e), e)[0].strip()
            })
            return

        # 2) Exec user code + test harness
        globs = {}
        try:
            exec(compiled, globs)
            exec(problem["test_code"], globs)
        except Exception:
            conn.send({
                "category": "runtime_error",
                "error": traceback.format_exc()
            })
            return

        # 3) Collect & run tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for obj in globs.values():
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                suite.addTests(loader.loadTestsFromTestCase(obj))

        output = io.StringIO()
        runner = unittest.TextTestRunner(stream=output, verbosity=0)
        result = runner.run(suite)
        if result.wasSuccessful():
            conn.send({"category": "success"})
        else:
            conn.send({
                "category": "test_failure",
                "error": output.getvalue().strip()
            })
    finally:
        conn.close()









def analyze_bigcodebench_problem(problem, completion,timeout=10):
    """
    Executes the user code + tests in a subprocess and classifies failures:
      - syntax_error
      - runtime_error
      - test_failure
      - segmentation_fault
      - timeout
      - success
    """
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(target=_worker, args=(problem, completion, child_conn))
    p.start()
    p.join(timeout)

    # Still alive? must've hung
    if p.is_alive():
        p.terminate()
        return {
            "category": "timeout",
            "error": f"Test execution timed out after {timeout} seconds"
        }

    # If worker sent us a result, use it
    if parent_conn.poll():
        return parent_conn.recv()

    # No result → process died unexpectedly. Check exit code
    exitcode = p.exitcode
    if exitcode is not None and exitcode < 0:
        sig = -exitcode
        # SIGSEGV is typically signal 11
        if sig == 11:
            cat = "segmentation_fault"
        else:
            cat = "crashed_by_signal"
        return {
            "category": cat,
            "error": f"Child process terminated by signal {sig}"
        }
    else:
        return {
            "category": "runtime_error",
            "error": f"Child process exited with code {exitcode} without reporting"
        }

# if __name__ == '__main__':
#     # testcase = "import matplotlib.pyplot as plt\nimport numpy as np\n\ndef task_func(ax, func_index):\n    if not isinstance(ax, matplotlib.axes.Axes):\n        raise ValueError(\"The input ax must be an instance of Axes\")\n    \n    func = FUNCTIONS[func_index]\n    theta = np.linspace(0, 2 * np.pi, 100)\n    r = func(np.sin(theta))\n    \n    ax.plot(theta, r, label=f'{func.__name__}')\n    ax.set_polar=True\n    ax.set_theta_zero_location('N')\n    ax.set_theta_direction(-1)\n    ax.set_thetagrids(range(0, 360, 45))\n    \n    return ax\n"
#     problems = load_bigcodebench()
#     first = problems[0]
#     print("Load sample:", first["task_id"], first["prompt"])
#     print("Test sample solution passes:", analyze_bigcodebench_problem(first, first["code_prompt"]+"\n"+first["canonical_solution"]))


