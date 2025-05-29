from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import transformers
import torch
import json
import copy
from typing import Any, Tuple
from tqdm import tqdm
import string
import loadcallnavi
import openai
from ttmg import TemplateFillingProcessor
import os
import re
import loadbfcl
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# modelprefix="llama3.1-8b-"

import itertools
eof="<|end_of_text|>"



def write_json_to_file(json_data, filename):
    with open(filename, 'a') as json_file:
        jstr=json.dumps(json_data)
        json_file.write(jstr+"\n")  # Add a newline after each JSON object
    json_file.close()

def writefile(l,dct):
    stra=json.dumps(dct)
    file=open(l+".json",'w')
    file.write(stra)
    file.close()

def generate_json_schema(obj: Any) -> dict:
    """
    Recursively generate a JSON Schema for a given JSON-like object.
    This basic implementation infers types from the sample object.
    """
    if isinstance(obj, dict):
        properties = {}
        required = []
        for key, value in obj.items():
            properties[key] = generate_json_schema(value)
            required.append(key)
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    elif isinstance(obj, list):
        if obj:
            # Assume homogeneous items; use the schema of the first item.
            return {"type": "array", "items": generate_json_schema(obj[0])}
        else:
            return {"type": "array"}
    elif isinstance(obj, int):
        return {"type": "integer"}
    elif isinstance(obj, float):
        return {"type": "number"}
    elif isinstance(obj, bool):
        return {"type": "boolean"}
    else:
        return {"type": "string"}

def json_to_regex(obj):
    if isinstance(obj, dict):
        parts = []
        for key, value in obj.items():
            key_pattern = r'\s*"' + re.escape(key) + r'"\s*:\s*'
            value_pattern = json_to_regex(value)
            parts.append(key_pattern + value_pattern)
        pattern = r'\{\s*' + r'\s*,\s*'.join(parts) + r'\s*\}'
        return pattern
    elif isinstance(obj, str):
        return r'"[^"]*"'
    elif isinstance(obj, (int, float)):
        return r'\d+'
    elif isinstance(obj, list):
        elements = [json_to_regex(item) for item in obj]
        pattern = r'\[\s*' + r'\s*,\s*'.join(elements) + r'\s*\]'
        return pattern
    else:
        return r'.*'


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

def mask_function_call(s: str) -> str:
    """
    Masks all argument values in a bracketed function-call string
    (quoted or unquoted) by replacing them with <BLANK>, without
    getting tripped up by commas inside quotes.
    """
    # strip outer [ ] and split off the args
    core = s.strip()[1:-1]  # drops leading '[' and trailing ']'
    fn_name, rest = core.split("(", 1)
    args_str = rest.rstrip(")")  # remove the trailing ')'
    
    # find all name=value pairs, where value is either
    #  - double-quoted string "…"
    #  - single-quoted string '…'
    #  - or unquoted chunk of non-comma chars
    assignments = re.findall(r'(\w+)=(".*?"|\'.*?\'|[^,]+)', args_str)
    
    # rebuild with all values masked
    masked = ", ".join(f"{name}=\"<BLANK>\"" for name, _ in assignments)
    return f"[{fn_name}({masked})]"


def bfcl_to_regex(template: str) -> str:
    """
    Turns a <BLANK>-masked template into an anchored regex with
    unnamed non-greedy groups (.*?) in place of each <BLANK>.
    """
    # split on the literal "<BLANK>"
    parts = template.split("<BLANK>")
    # escape each literal chunk, and between them inject a (.*?) group
    regex_parts = []
    for i, part in enumerate(parts):
        regex_parts.append(re.escape(part))
        if i < len(parts) - 1:
            regex_parts.append("(.*?)")
    
    # join, then anchor
    pattern = "^" + "".join(regex_parts) + "$"
    return pattern



def wildcard_compare(val1: Any, val2: Any) -> bool:
    """
    Recursively compare two normalized JSON values.
    If either value is the wildcard string "$$$", consider it a match.
    """
    # If either value is the wildcard string, they match.
    if isinstance(val1, str) and val1 == "$$$":
        return True
    if isinstance(val2, str) and val2 == "$$$":
        return True

    # If both are dicts, compare keys (order ignored) and then values.
    if isinstance(val1, dict) and isinstance(val2, dict):
        if set(val1.keys()) != set(val2.keys()):
            return False
        for k in val1.keys():
            if not wildcard_compare(val1[k], val2[k]):
                return False
        return True

    # If both are lists, compare ignoring order.
    if isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return False
        # Create a copy of list2 items to track matches.
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

    # Otherwise, compare using equality.
    return val1 == val2

def compare_json_with_wildcard(json_obj1: Any, json_obj2: Any) -> bool:
    """
    Compare two JSON objects (or Python dicts) by normalizing them (ignoring object order, case,
    punctuation, and whitespace) and treating any value equal to "$$$" as a wildcard match.
    """
    norm1 = normalize_json(json_obj1)
    norm2 = normalize_json(json_obj2)
    return wildcard_compare(norm1, norm2)

def replace_values_with_blank(obj: Any) -> Any:
    """
    Recursively replace all non-container values with "<BLANK>",
    except for the key "API" where, if its value is a list,
    we keep its first element unchanged.
    """
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            if key == "API" and isinstance(value, list):
                # Keep the first element as is (if exists), blank out the rest.
                if value:
                    new_list = [value[0]]
                    # For any additional elements, process recursively.
                    for elem in value[1:]:
                        new_list.append(replace_values_with_blank(elem))
                    new_dict[key] = new_list
                else:
                    new_dict[key] = value
            else:
                new_dict[key] = replace_values_with_blank(value)
        return new_dict
    elif isinstance(obj, list):
        return [replace_values_with_blank(item) for item in obj]
    else:
        # For any primitive value, return the blank token.
        return "<BLANK>"

def convert_int_to_string(obj: Any) -> Any:
    """
    Recursively convert all int values in the object to strings.
    """
    if isinstance(obj, dict):
        return {key: convert_int_to_string(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int_to_string(item) for item in obj]
    elif isinstance(obj, int):
        return str(obj)
    else:
        return obj

def transform_json(input_data: dict) -> Tuple[str, str]:
    """
    Given an input JSON (as a dict), returns a tuple of two JSON strings:
      1. All values replaced with "<BLANK>" (except the first API value remains unchanged)
      2. All integer values are converted to strings (other values unchanged)
    """
    blank_version = replace_values_with_blank(input_data)
    int_string_version = convert_int_to_string(input_data)
    
    # Return the JSON strings (optionally with indent=2 for readability)
    return json.dumps(blank_version), json.dumps(int_string_version)




def test_ttmt(model_name,output):
    modelprefix=model_name.split("/")[1]
    callnavi= loadcallnavi.loadcallnavi()

    tokenizer = AutoTokenizer.from_pretrained(model_name,device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype=torch.float16)

    if '<BLANK>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<BLANK>']})
        model.resize_token_embeddings(len(tokenizer))
    transformers.logging.set_verbosity_error()
    prompt_template = """Please fill the JSON template for API, and only output the completed unformatted JSON for the following question:'"""

    calc=0
    with open(output+"/"+modelprefix+"-ttmg-callnavi.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(callnavi), 1)):
        question = callnavi[i]["question"]
        api_list = callnavi[i]["API"]
        schema = callnavi[i]["schema"]
        answer = callnavi[i]["answer"]
        tmp,gt=transform_json(answer)
        
        temp=json.dumps(tmp)+"\n"+eof
        prompt_1='''
Give the API list with describtion below, then give the question in chatbot, please give me the correct API that should be called. 
=======API JSON Template start=======
        '''
        prompt_2='''
        \n=======API JSON Template end======= \n\n=======Question start=======
        '''
        prompt_3='''\n=======Question end=======
Given the user question, and the APIs, classify and give a correct API name and parameters to call. \n
Answer should be formatted includes API names and parameters in JSON style, looks like :
{"API": ["getCustomerDetails", "depositFunds"], "parameters":[{"parameter1ForCall1": "..." },{"parameter1ForCall2": "...", "parameter2ForCall2": "..."}]}

If we cannot get some parameter information from the question, set these parameters to "$$$".

NO explanation/notes in answer! Only JSON!
"
        '''

        system="""You are an expert API generator. Your task is to write JSON code based on the question and API provided."""     
        prompt = prompt_1+str(temp)+prompt_2+str(question)+prompt_3
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ones_like(input_ids)
        segments = temp.split("<BLANK>")
        # Convert each fixed segment to token IDs.
        forced_segments = [tokenizer.encode(seg, add_special_tokens=False) for seg in segments]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        processor = TemplateFillingProcessor(prompt_length=len(prompt_ids), forced_segments=forced_segments)
        logits_processor = LogitsProcessorList([processor])
        # 8. Generate output.
        # Set max_length to cover the prompt and enough tokens for the filled template.
        max_length = len(input_ids[0]) + 256
        device = torch.device("cuda")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor,
            max_length=max_length,
            do_sample=True,       # Use sampling to let blanks be generated freely.
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[128001, 128009]
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # result = generated_text[len(prompt.replace('<BLANK>','').replace('<|end_of_text|>',"")):].strip()
        result =generated_text.split("NO explanation/notes in answer! Only JSON!\n\"")[-1].strip()
        result = result.split("\n<|end_of_text|>")[0].replace("Based","").replace("{\"\"","").strip()
        # result = generated_text.replace(prompt_1,"").replace(str(temp),"").replace(prompt_2,"").replace(str(question),"").replace(prompt_3,"").replace(json.dumps(tmp).replace("<BLANK>",""),"").strip()
        # prompt = prompt_1+str(api_list)+prompt_2+str(question)+prompt_3
        # if decoder=="xgrammar":
        #         result = test_model_ebnf(model, prompt,system,json_stmt)
        # else:
        #         result = test_model_ebnf(model, prompt,system,json_lark)
        # result = test_model(model, prompt,system)

        if "}]}" in result:
            result = result.split("}]}")[0]+"}]}"
        test_result=loadcallnavi.test_callnavi(json.dumps(answer).replace("'","\""), result)
        if not test_result[0]:
            print(generated_text)
        # print("Result:", result)
        # print("Ground Truth:", answer)
        # print("Test Result:", test_result)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "gt":json.dumps(answer),"match":test_result}, output+"/"+modelprefix+"-ttmg-callnavi.jsonl")



def test_model(client,model_name, prompt,system):
    try:
        if "gemma" not in model_name:

            response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]),
            # request_timeout=15,  # Set a timeout for the request
        else:
            response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": system+"\n"+prompt},
            ],
            # request_timeout=15,  # Set a timeout for the request
            )
        generated=response[0].choices[0].message.content
        # generated = completion
    except Exception as e:
        print(response)
        generated=""
        print(f"Error in test_model: {e}")
    return generated
def test_model_regex(client,model_name, prompt,system, pattern):

    try:
        if "gemma" not in model_name:

            completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            extra_body={"regex": pattern},
            # request_timeout=15,  # Set a timeout for the request
            )
            
        else:
            completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "user", "content": system+"\n"+prompt},
            ],
            extra_body={"regex": pattern},
            # request_timeout=15,  # Set a timeout for the request
            )
        generated=completion.choices[0].message.content
    except Exception as e:
        generated=""
    return generated

def test_model_js(model_name, prompt,system,schema,client):

    try:
        completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={
        "type": "json_schema",
        "json_schema": {"name": "foo", "schema":schema}},
        # request_timeout=15,  # Set a timeout for the request
    )
        generated=completion.choices[0].message.content

    except Exception as e:
        generated='[]'
    return generated

def test_with_jsonschema(client,model_name,output):
    modelprefix=model_name.split("/")[1]
    callnavi= loadcallnavi.loadcallnavi()

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
        jsonschema = generate_json_schema(answer)
        pattern= json_to_regex(answer)
        # print("pattern:", pattern)
        # result = test_model_js(model_name, prompt,system,jsonschema,client)
        # result = test_model(client,model_name, prompt,system)

        result = test_model_regex(client,model_name, prompt,system, pattern)

        test_result=loadcallnavi.test_callnavi(json.dumps(answer), result)
        # print("Result:", result)
        # print("Ground Truth:", answer)
        # print("Test Result:", test_result)
        write_json_to_file({"result":result, "gt":json.dumps(answer),"match":test_result}, output+"/"+modelprefix+"-regex-callnavi.jsonl")


def test_jsonschema(model_name,output):
    from sglang.test.test_utils import is_in_ci

    if is_in_ci():
        from patch import launch_server_cmd
    else:
        from sglang.utils import launch_server_cmd

    from sglang.utils import wait_for_server, print_highlight, terminate_process

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # "python -m sglang.launch_server --model-path "+model+" --host 0.0.0.0 --grammar-backend llguidance outlines"

    server_process, port = launch_server_cmd(
        "python -m sglang.launch_server --model-path "+model_name+" --host 0.0.0.0 --mem-fraction-static 0.8 --tp 2",
    )

    wait_for_server(f"http://localhost:{port}")
    client = openai.Client(base_url=f"http://0.0.0.0:{port}/v1", api_key="None",timeout=15)

    with open(output+"/"+modelprefix+"-ttmg-callnavi.jsonl", "w") as f:
        f.write("\n")
    f.close()
    test_with_jsonschema(client,model_name,output)



def generate_ground_truth(specs):
    """
    Given a list of mappings from function names to parameter-value lists,
    produce all concrete function-call strings (ground truth).
    """
    calls = []
    for spec in specs:
        for func, params in spec.items():
            # Extract parameter names and their possible values
            keys, value_lists = zip(*params.items())
            # Produce Cartesian product of all values
            for combination in itertools.product(*value_lists):
                # Format each argument as key="value"
                args = ", ".join(f'{k}="{v}"' for k, v in zip(keys, combination))
                calls.append(f"[{func}({args})]")
    return calls
fcpattern = '\[?\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(\s*(.*)\)\s*\]?'

def test_bfcl_v2_ast(client,model_name, output):
    modelprefix=model_name.split("/")[1]

    bfcl=loadbfcl.load_bfcl_v2_ast_dataset()
    with open(output+"/"+modelprefix+"-regex-bfcl.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(bfcl), 1)):
        problem = bfcl[i]
        prompt = problem["question"]
        system = problem["prompt"]
        ground_truth = problem["ground_truth"]
        try:
            gt=generate_ground_truth(json.loads(ground_truth))
            # print("Ground Truth:", gt)
            if gt != []:
                masked=mask_function_call(gt[0])
            bfclpattern=bfcl_to_regex(masked).replace("^","").replace("$","")
        except Exception as e:
            print("Error generating ground truth:", e)
            bfclpattern=fcpattern
            continue
        print("bfclpattern:", bfclpattern)
        result = test_model_regex(client=client, model_name=model_name, system="""You are a helpful assistant and an expert in function composition. Just follow the format in the prompt, output should be formatted function calling start with a "[".""", prompt=system+"\nThe Question is: "+prompt+"\nThe function call template is: "+masked+"========END of TEMPLATE========\n", pattern=bfclpattern)
        # +"\nThe function call template is: "+masked
        # result = test_model(client=client, model_name=model_name, system="""You are a helpful assistant and an expert in function composition. Just follow the format in the prompt, output should be formatted function calling start with a "[".""", prompt=system+"\nThe Question is: "+prompt+"\nThe function call template is: "+masked+"========END of TEMPLATE========\n",)
        if ")]" in result:
            result = result.split(")]")[0]+")]"
        print("Output:", result)
        # if "========END of TEMPLATE========" in result:
        #     result = result.split("========END of TEMPLATE========")[0].strip()
        test_result=loadbfcl.test_bfcl_v2_ast(result, ground_truth)
        # print("Ground Truth:", ground_truth)
        # print("Result:", test_result)
        write_json_to_file({"result":result, "match":test_result}, output+"/"+modelprefix+"-regex-bfcl.jsonl")


# def ttmg_bfcl():
#     tokenizer = AutoTokenizer.from_pretrained(model_name,device_map="auto")
#     model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype=torch.float16)

#     if '<BLANK>' not in tokenizer.get_vocab():
#         tokenizer.add_special_tokens({'additional_special_tokens': ['<BLANK>']})
#         model.resize_token_embeddings(len(tokenizer))
#     transformers.logging.set_verbosity_error()

def test_bfcl(model_name, output):
    from sglang.test.test_utils import is_in_ci

    if is_in_ci():
        from patch import launch_server_cmd
    else:
        from sglang.utils import launch_server_cmd

    from sglang.utils import wait_for_server, print_highlight, terminate_process

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # "python -m sglang.launch_server --model-path "+model+" --host 0.0.0.0 --grammar-backend llguidance outlines"

    server_process, port = launch_server_cmd(
        "python -m sglang.launch_server --model-path "+model_name+" --host 0.0.0.0 --mem-fraction-static 0.8 --tp 2",
    )

    wait_for_server(f"http://localhost:{port}")
    client = openai.Client(base_url=f"http://0.0.0.0:{port}/v1", api_key="None",timeout=15)
    test_bfcl_v2_ast(client,model_name, output)



def test_bfcl_ttmt(model_name, output):
    modelprefix=model_name.split("/")[1]

    tokenizer = AutoTokenizer.from_pretrained(model_name,device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype=torch.float16)
    bfcl=loadbfcl.load_bfcl_v2_ast_dataset()
    if '<BLANK>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<BLANK>']})
        model.resize_token_embeddings(len(tokenizer))
    transformers.logging.set_verbosity_error()

    calc=0
    with open("ttmg/"+modelprefix+"-ttmg-bfcl.jsonl", "w") as f:
        f.write("\n")
    f.close()
    system_msg="""You are a helpful assistant and an expert in function composition. Just follow the format in the prompt, output should be formatted function calling start with a "[]"."""
    for i in tqdm(range(0, len(bfcl), 1)):
        problem = bfcl[i]
        prompt = problem["question"]
        system = problem["prompt"]
        ground_truth = problem["ground_truth"]
        try:
            gt=generate_ground_truth(json.loads(ground_truth))
            if gt != []:
                masked=mask_function_call(gt[0])+eof
        except Exception as e:
            print("Error generating ground truth:", e)
            masked="[<BLANK>]"+eof
            continue
        # print("Masked Function Call Template:", masked)
        prompt= system_msg+system+"\nThe Question is: "+prompt+"\nThe function call template is: "+masked+"========END of TEMPLATE========\n"

        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ones_like(input_ids)
        segments = masked.split("<BLANK>")
        # Convert each fixed segment to token IDs.
        forced_segments = [tokenizer.encode(seg, add_special_tokens=False) for seg in segments]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        processor = TemplateFillingProcessor(prompt_length=len(prompt_ids), forced_segments=forced_segments)
        logits_processor = LogitsProcessorList([processor])
        # 8. Generate output.
        # Set max_length to cover the prompt and enough tokens for the filled template.
        max_length = len(input_ids[0]) + 256
        device = torch.device("cuda:0")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor,
            max_length=max_length,
            do_sample=True,       # Use sampling to let blanks be generated freely.
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[128001, 128009]
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("Generated Text:", generated_text)
        result=generated_text.split("========END of TEMPLATE========")[1].replace("<|end_of_text|>","").strip()
        if ")]" in result:
            result = result.split(")]")[0]+")]"
        result = "["+result.split("[")[-1]
        print("Result:", result)
        test_result=loadbfcl.test_bfcl_v2_ast(result, ground_truth)
        write_json_to_file({"result":result, "match":test_result}, "ttmg/"+modelprefix+"-ttmg-bfcl.jsonl")


model_name = "Qwen/Qwen2.5-7B-Instruct"
modelprefix="qwen-2.5-"
# eof="<|endoftext|>"

def run_ttmg(model, task, decoder, output):
    if task == "callnavi":
        if decoder=="ttmg":
            test_ttmt(model,output)
        else:
            test_bfcl_v2_ast(model,output)
    elif task=="bfcl":
        if decoder=="ttmg":
            test_bfcl_ttmt(model,output)
        else:
            test_bfcl(model,output)
    else:
        raise ValueError("Unknown task")