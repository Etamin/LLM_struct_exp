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
import openai
import os
from sglang.test.test_utils import is_in_ci


mode="sglang"

decoder="outlines"



import os





# mode="openai"




def write_json_to_file(json_data, filename):
    filename="results/"+filename
    with open(filename, 'a') as json_file:
        jstr=json.dumps(json_data)
        json_file.write(jstr+"\n")  # Add a newline after each JSON object
    json_file.close()


def test_model_js(client,model_name, prompt,system,schema):

    try:
        if "gemma" not in model_name:
            completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            extra_body={"guided_json": schema},
        )
            generated=completion.choices[0].message.content
        else:
            completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "user", "content": system+"\n"+prompt},
            ],
            extra_body={"guided_json": schema},
            )
        generated=completion.choices[0].message.content
    except Exception as e:
        generated='[]'
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
            extra_body={"regex": bfclpattern},
            )
            
        else:
            completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "user", "content": system+"\n"+prompt},
            ],
            extra_body={"regex": bfclpattern},
            )
        generated=completion.choices[0].message.content
    except Exception as e:
        generated=""
    return generated

def test_model_ebnf(client,model_name, prompt, system,ebnf):
    try:
        if decoder=="outlines" or decoder=="llguidance":
            ext={"guided_grammar": ebnf}
        else:
            ext={"ebnf": ebnf}
        
        if "gemma" not in model_name:

            completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[            
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            extra_body=ext,
            )
            generated=completion.choices[0].message.content
        else:
            completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[            
                {"role": "user", "content": system+"\n"+prompt},
            ],
            extra_body=ext,
            )
            generated=completion.choices[0].message.content
    except Exception as e:
        generated=""
    return generated








def test_spider(client, model,modelprefix,output,decoder):
    with open("Grammars/sql.lark", "r") as f:
        sql_lark = f.read()
    spider=loadspider.load_cm_spider_dataset()

    calc=0
    with open(output+"/"+modelprefix+"spider.jsonl", "w") as f:
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
        result = test_model_ebnf(client,model, prompt,system,sql_lark)
        # result = test_model(model, prompt,system)

        test_result=loadspider.execute_query_on_schema(problem["schema"], result, problem["query"])
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "match":test_result}, output+"/"+modelprefix+"spider.jsonl")
    return (calc,len(spider))


def test_callnavi(client, model,modelprefix,output,decoder):
    callnavi=loadcallnavi.loadcallnavi()
    with open("Grammars/json.gbnf", "r") as f:
        json_stmt = f.read()
    with open("Grammars/json.lark", "r") as f:
        json_lark = f.read()

    calc=0
    with open(output+"/"+modelprefix+"callnavi.jsonl", "w") as f:
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
        if decoder=="xgrammar":
                result = test_model_ebnf(client,model, prompt,system,json_stmt)
        else:
                result = test_model_ebnf(client,model, prompt,system,json_lark)
        # result = test_model(model, prompt,system)

        test_result=loadcallnavi.test_callnavi(json.dumps(answer), result)
        # print("Result:", result)
        # print("Ground Truth:", answer)
        # print("Test Result:", test_result)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "gt":json.dumps(answer),"match":test_result}, output+"/"+modelprefix+"callnavi.jsonl")
    return (calc,len(callnavi))

def test_bfcl_v2_ast(client, model,modelprefix,output,decoder):
    bfclpattern = '\[?\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(\s*(.*)\)\s*\]?'

    bfcl=loadbfcl.load_bfcl_v2_ast_dataset()

    calc=0
    with open(output+"/"+modelprefix+"bfcl_v2_ast.jsonl", "w") as f:
        f.write("\n")
    f.close()
    for i in tqdm(range(0, len(bfcl), 1)):
        problem = bfcl[i]
        prompt = problem["question"]
        system = problem["prompt"]
        ground_truth = problem["ground_truth"]

        result = test_model_regex(client,model, """You are a helpful assistant and an expert in function composition. Just follow the format in the prompt, output should be formatted function calling start with a "[]".""", system+"\nThe Question is: "+prompt, bfclpattern)
        test_result=loadbfcl.test_bfcl_v2_ast(result, ground_truth)
        if test_result:
            calc+=1
        write_json_to_file({"result":result, "match":test_result}, output+"/"+modelprefix+"bfcl_v2_ast.jsonl")
    return (calc,len(bfcl))

def write_timestamp_to_file(filename, task_name):
    with open(filename, 'a') as f:
        f.write("\n")
        f.write(task_name+": Timestamp: " + str(datetime.datetime.now()) + "\n")
    f.close()

def test_stab():
    for i in range(5):
        print("BFCL:")
        test_spider(i)

def run_outlines(model,task, method, decoder, output):
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
            "python -m sglang.launch_server --model-path "+model+" --host 0.0.0.0 --mem-fraction-static 0.8 --grammar-backend "+decoder,
        )

        wait_for_server(f"http://localhost:{port}")
        client = openai.Client(base_url=f"http://0.0.0.0:{port}/v1", api_key="None",timeout=15)

    if mode=="openai":
        os.environ["OPENAI_API_KEY"]=""
        client = OpenAI()

    modelprefix=model.split("/")[1]
    modelprefix+=decoder+"-"
    if task=="callnavi":
        test_callnavi(client, model,modelprefix,output,decoder)
    elif task=="bfcl":
        test_bfcl_v2_ast(client, model,modelprefix,output,decoder)
    elif task=="spider":
        test_spider(client, model,modelprefix,output,decoder)
    else:
        raise ValueError("No supported task input")
