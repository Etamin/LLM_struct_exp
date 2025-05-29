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
model = "meta-llama/Llama-3.1-8B-Instruct"
modelprefix="llama-3.1-8b--"
# model = "google/gemma-2-9b-it"
# modelprefix="gemma-2-9b-test-"
import os

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
# model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
# modelprefix="llama-3.1-70b--"
# model = "Qwen/Qwen2.5-7B-Instruct"
# modelprefix="qwen-2.5-outlines-"
# model="gpt-4.1-mini"
# modelprefix="gpt-4.1-mini-"
import openai
import os
from sglang.test.test_utils import is_in_ci



# mode="openai"
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

if mode=="openai":
    os.environ["OPENAI_API_KEY"]=""
    client = OpenAI()

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

def test_model_js(model_name, prompt,system,schema):

    try:
        completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        extra_body={"guided_json": schema},
    )
        generated=completion.choices[0].message.content

    except Exception as e:
        generated='[]'
    return generated


def test_model_ebnf(model_name, prompt, system,ebnf):
    try:
        completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[            
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        extra_body={"guided_grammar": ebnf},
        # extra_body={"ebnf": ebnf},
        )
        generated=completion.choices[0].message.content
    except Exception as e:
        generated=""
    return generated

with open("Grammars/sql.lark", "r") as f:
    sql_lark = f.read()
with open("Grammars/python.lark", "r") as f:
    py_lark = f.read()
with open("Grammars/json.gbnf", "r") as f:
    json_stmt = f.read()
with open("Grammars/json.lark", "r") as f:
    json_lark = f.read()

humaneval=loadhumaneval.load_humaneval()
xlam=loadxlam.load_xlam_dataset()
callnavi=loadcallnavi.loadcallnavi()
bigbench=loadbigbench.load_bigcodebench()
spider=loadspider.load_cm_spider_dataset()
bfcl=loadbfcl.load_bfcl_v2_ast_dataset()
# Test the model with different datasets
# Test humaneval

def test_humaneval():
    calc=0
    with open(modelprefix+"humaneval.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(humaneval), 1)):
        problem = humaneval[i]
        system="You are a helpful AI assistant specialized in Python code generation."
        prompt = """Complete the following Python function. Only provide the Python code that completes the function.
Do not add any explanations, comments outside the function body, or any text other than the code itself.

```python\n"""+problem["prompt"]
        result = test_model(model, prompt,system)
        test_result=loadhumaneval.test_humaneval_problem(problem, result)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "match":test_result}, modelprefix+"humaneval.jsonl")
    return (calc,len(humaneval))


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

def test_bigcodebench_canvas():
    calc=0
    with open(modelprefix+"bigcodebench_canvas.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(bigbench), 1)):
        problem = bigbench[i]
        prompt = f"""
Given:
- Instruction:
{problem["prompt"]}

Task:
Generate Python code that fulfills the above instruction.

Input:
The instruction text provided.

Output:
Only the necessary Python code to solve the task (no explanations, comments, or boilerplate)."""
        
        system="""You are an AI programming assistant. Your task is to write Python code based on the instructions and context provided.
Focus on generating only the necessary Python code to fulfill the request.
Do not include any surrounding text, explanations, or the original boilerplate unless it's part of the required completion.
"""
        result = test_model(model, prompt,system)
        test_result=loadbigbench.analyze_bigcodebench_problem(problem, result)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "match":test_result}, modelprefix+"bigcodebench_canvas.jsonl")

    return (calc,len(bigbench))


def test_spider():
    calc=0
    with open(modelprefix+"spider.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(spider), 1)):
        problem = spider[i]
        prompt = "Schema definitions:\n"+problem["create"]+\
            "\n\nQuestion: "+problem["question"]+\
            "\n\nSQL Query:"
        system="""You are an expert SQL generator. assistant. Your task is to write SQL code based on the question and database schema provided.
Focus on generating only the necessary SQL code to fulfill the request.
Do not include any surrounding text, explanations, or the original boilerplate unless it's part of the required completion.
"""
        # result = test_model_ebnf(model, prompt,system,sql_lark)
        result = test_model(model, prompt,system)

        test_result=loadspider.execute_query_on_schema(problem["schema"], result, problem["query"])
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "match":test_result}, modelprefix+"spider.jsonl")
    return (calc,len(spider))

def test_spider_canvas():
    calc=0
    with open(modelprefix+"spider_canvas.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(spider), 1)):
        problem = spider[i]
        prompt = f"""
Given:
- Schema definitions:
{problem["create"]}
- Question:
{problem["question"]}

Task:
Generate a single SQL query that answers the question using the provided schema.

Input:
The schema definitions and question above.

Output:
A valid SQL query (only the query, no explanations or comments).
"""
        system="""You are an expert SQL generator. assistant. Your task is to write SQL code based on the question and database schema provided.
Focus on generating only the necessary SQL code to fulfill the request.
Do not include any surrounding text, explanations, or the original boilerplate unless it's part of the required completion.
"""
        result = test_model(model, prompt,system)
        test_result=loadspider.execute_query_on_schema(problem["schema"], result)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "match":test_result}, modelprefix+"spider_canvas.jsonl")
    return (calc,len(spider))


def test_xlam():
    calc=0
    with open(modelprefix+"xlam.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(xlam), 1)):
        query= xlam[i]["query"]
        tools = xlam[i]["tools"]
        prompt = f"""
Given an input question for a function call, you are required to generate the JSON code for the function call.
\"Question\": \"{query}\",        
Please respond with a JSON object of the form:\n
\"Tools\": {tools}

Output:
"""
        system="""You are a function‐calling agent. Your task is to write JSON code based on the question and API provided."""
        result = test_model(model, prompt,system)
        test_result=jsoncompare.compare_json_with_wildcard(json.dumps(xlam[i]["answers"]), result)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "match":test_result}, modelprefix+"xlam.jsonl")
    return (calc,len(xlam))


def test_xlam_canvas():
    calc=0
    with open(modelprefix+"xlam_canvas.jsonl", "w") as f:
        f.write("\n")
    f.close()

    for i in tqdm(range(0, len(xlam), 1)):
        query= xlam[i]["query"]
        tools = xlam[i]["tools"]
        prompt = f"""You are a function‐calling agent. Given an input JSON containing:
1. A natural‐language `query`.
2. A list of available `tools`, each with:
   • `name` (string)  
   • `description` (string)  
   • `parameters`: a JSON object mapping each parameter name to an object with `type`, `description`, and `required` fields.

Your task is to output _only_ a JSON array of calls, where each element has:
- `"name"`: the tool’s name  
- `"arguments"`: an object mapping parameter names to their chosen values  

Follow these rules:
- Include only the functions needed to answer the query.  
- Do _not_ include any text outside the JSON array (no explanations, no extra fields).  
- Ensure parameter values match the specified types.

**Input:**  
```json
{{
  "query": "{query}",
  "tools": {tools}
}}"""
        system="""You are a function‐calling agent. Your task is to write JSON code based on the question and API provided."""
        result = test_model(model, prompt,system)
        test_result=jsoncompare.compare_json_with_wildcard(json.dumps(xlam[i]["answers"]), result)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "match":test_result}, modelprefix+"xlam_canvas.jsonl")
    return (calc,len(xlam))

# test callnavi without canvas
def test_callnavi():
    calc=0
    with open(modelprefix+"callnavi.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(callnavi), 1)):
        question = callnavi[i]["question"]
        api_list = callnavi[i]["API"]
        schema = callnavi[i]["schema"]
        answer = callnavi[i]["answer"]
        prompt_1='''
        Give the API list with describtion below, then give the question in chatbot, please give me the correct API that should be called. 
        =======list start=======
        '''
        prompt_2='''
        \n=======JSON end======= \n\n=======Question start=======
        '''
        prompt_3='''\n=======Question end=======
        Given the user question, and the APIs, classify and give a correct API name and parameters to call. \n
        Answer should be formatted includes API names and parameters in JSON style, looks like :
        {'API': ['getCustomerDetails', 'depositFunds'], 'parameters':[{"parameter1ForCall1": "***" },{"parameter1ForCall2": "***", "parameter2ForCall2": "***"}]}


        If we cannot get some parameter information from the question, set these parameters to "$$$".
        
        NO explanation/notes in answer! Only JSON!"
        '''

        system="""You are an expert API generator. Your task is to write JSON code based on the question and API provided.
Focus on generating only the necessary JSON code to fulfill the request.    
Do not include any surrounding text, explanations, or the original boilerplate unless it's part of the required completion.
"""     
        prompt = prompt_1+str(api_list)+prompt_2+str(question)+prompt_3
        # result = test_model_ebnf(model, prompt,system,json_lark)
        result = test_model(model, prompt,system)

        test_result=loadcallnavi.test_callnavi(json.dumps(answer), result)
        # print("Result:", result)
        # print("Ground Truth:", answer)
        # print("Test Result:", test_result)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "gt":json.dumps(answer),"match":test_result}, modelprefix+"callnavi.jsonl")
    return (calc,len(callnavi))

# Test callnavi with canvas
def test_callnavi_canvas():
    calc=0
    with open(modelprefix+"callnavi_canvas.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(callnavi), 1)):
        question = callnavi[i]["question"]
        api_list = callnavi[i]["API"]
        schema = callnavi[i]["schema"]
        answer = callnavi[i]["answer"]
        prompt = f"""You are an API‐calling assistant.  
Given:
1. A user question in a chat turn.
2. A list of available APIs (each with name, description, parameters).
3. A JSON Schema for each API (defining types and required fields).

Your task:
- Select the minimal set of APIs needed.
- Output _only_ a JSON object with two keys:
  • "API": a list of chosen API names  
  • "parameters": an object mapping each required parameter to its value  

Do not include any explanation or additional fields—produce only the JSON.

Input:
Question:
{question}

APIs:
{api_list}

Output Format:
{{"API": ["APIName1",...], "parameters": [{{...}},...]}}"""
        system="""You are an expert API generator. Your task is to write JSON code based on the question and API provided.
Focus on generating only the necessary JSON code to fulfill the request.    
Do not include any surrounding text, explanations, or the original boilerplate unless it's part of the required completion.
"""
        result = test_model_js(model, prompt,system,schema)
        test_result=loadcallnavi.test_callnavi(json.dumps(answer), result)
        print(test_result)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "match":test_result}, modelprefix+"callnavi_canvas.jsonl")

    return (calc,len(callnavi))

def test_bfcl_v2_ast():
    calc=0
    with open(modelprefix+"bfcl_v2_ast.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(bfcl), 1)):
        problem = bfcl[i]
        prompt = problem["question"]
        system = problem["prompt"]
        ground_truth = problem["ground_truth"]
        result = test_model(model, """You are a helpful assistant and an expert in function composition. Just follow the format in the prompt, output should be plain text, without any "```" or "```python"/"```json" wraps.""", system+"\nThe Question is: "+prompt)
        test_result=loadbfcl.test_bfcl_v2_ast(result, ground_truth)
        # print("Output:", result)
        # print("Ground Truth:", ground_truth)
        # print("Result:", test_result)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "match":test_result}, modelprefix+"bfcl_v2_ast.jsonl")
    return (calc,len(bfcl))

def write_timestamp_to_file(filename, task_name):
    with open(filename, 'a') as f:
        f.write("\n")
        f.write(task_name+": Timestamp: " + str(datetime.datetime.now()) + "\n")
    f.close()

timelog_file = "timelog.txt"
# print("HumanEval, Non-Canvas:")
# print(test_humaneval())
# print("BigCodeBench, Non-Canvas:")
# write_timestamp_to_file(timelog_file, modelprefix+"BigCodeBench "+" "+"Start")
# print(test_bigcodebench())
# write_timestamp_to_file(timelog_file, modelprefix+"BigCodeBench "+" "+"End")
# print("BigCodeBench, Canvas:")
# print(test_bigcodebench_canvas())
# print("Spider, Non-Canvas:")
# write_timestamp_to_file(timelog_file, modelprefix+"Spider "+" "+"Start")
# print(test_spider())
# write_timestamp_to_file(timelog_file, modelprefix+"Spider "+" "+"End")
# print("Spider, Canvas:")
# print(test_spider_canvas())
# print("XLAM, Non-Canvas:")
# print(test_xlam())
# print("XLAM, Canvas:")
# print(test_xlam_canvas())
# write_timestamp_to_file(timelog_file, modelprefix+"CallNavi "+" "+"Start")
# print("CallNavi, Non-Canvas:")
# print(test_callnavi())
# write_timestamp_to_file(timelog_file, modelprefix+"CallNavi "+" "+"End")
# print("CallNavi, Canvas:")
# print(test_callnavi_canvas())
print("BFCL, Non-Canvas:")
write_timestamp_to_file(timelog_file, modelprefix+"bfcl "+" "+"Start")
print(test_bfcl_v2_ast())
write_timestamp_to_file(timelog_file, modelprefix+"bfcl "+" "+"End")