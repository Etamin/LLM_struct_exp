from openai import OpenAI
import json
import loadbigbench
import loadcallnavi
import loadxlam
import loadhumaneval
import loadspider
import loadbfcl
from tqdm import tqdm
import jsoncompare
import datetime
import os
from openai import OpenAI
import openai
import os
from sglang.test.test_utils import is_in_ci
model = "meta-llama/Llama-3.1-8B-Instruct"
modelprefix="llama-3.1-8b--"

mode="sglang"

if mode=="sglang":
    if is_in_ci():
        from patch import launch_server_cmd
    else:
        from sglang.utils import launch_server_cmd

    from sglang.utils import wait_for_server, print_highlight, terminate_process

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # "python -m sglang.launch_server --model-path "+model+" --host 0.0.0.0 --grammar-backend llguidance outlines"

    server_process, port = launch_server_cmd(
        "python -m sglang.launch_server --model-path "+model+" --host 0.0.0.0 --tp 2 --mem-fraction-static 0.8",
    )

    wait_for_server(f"http://localhost:{port}")
    client = openai.Client(base_url=f"http://0.0.0.0:{port}/v1", api_key="None")

def write_json_to_file(json_data, filename):
    with open(filename, 'a') as json_file:
        jstr=json.dumps(json_data)
        json_file.write(jstr+"\n")  # Add a newline after each JSON object
    json_file.close()

def test_model(model_name, prompt,system):

    try:
        if "gemma" not in model_name:

            completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ])
        else:
            completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "user", "content": system+"\n"+prompt},
            ]
            )
        generated=completion.choices[0].message.content
    except Exception as e:
        generated=""
    return generated


bigbench=loadbigbench.load_bigcodebench()


def test_bigcodebench():
    calc=0
    with open(modelprefix+"bigcodebench.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(bigbench), 1)):
        problem = bigbench[i]
        prompt = problem["prompt"]
        system="""You are an AI programming assistant. Your task is to write Python code based on the instructions and context provided.
Focus on generating only the necessary Python code to fulfill the request.
Do not include any surrounding text, explanations, or the original boilerplate unless it's part of the required completion.
"""
        # result = test_model_ebnf(model, prompt,system,py_lark)
        result = test_model(model, prompt,system)
        test_result=loadbigbench.analyze_bigcodebench_problem(problem, result)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "match":test_result}, modelprefix+"bigcodebench.jsonl")
    return (calc,len(bigbench))

print(test_bigcodebench())
