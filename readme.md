

It includes implementations and benchmarks for grammar-based decoding, regex constraints, and our proposed **Template Token Match Generation (TTMG)** method across tasks such as JSON generation, function calling, and SQL query construction.

---

## 🔍 Features

- **Benchmark Tasks**:  
  - `CallNavi`: JSON-based structured API call generation  
  - `BFCL`: Function calling with strict schema  
  - `Spider`: SQL query generation  
  - `BigCodeBench`: General code generation

- **Structure Control Methods**:
  - Unconstrained decoding  
  - Regex-constrained decoding  
  - JSON Schema-based decoding (via XGrammar)  
  - Template Token Match Generation (TTMG)

- **Multi-backend support**: Works with [SGLang](https://github.com/InternLM/SGLang)

- **Metrics & Error Classification**:
  - Syntax, structural, and value error detection
  - AST match and exact match scores
  - Structured error type breakdown and visualization

---
## Install dependencies on Python 3.10+
```bash
pip install -r requirements.txt
pip install -r bigcodebench_eval_requirements.txt
git clone https://github.com/Etamin/CallNavi.git
```


## Running Experiments

```bash
python run.py --task callnavi --model Qwen/Qwen2.5-7B-Instruct --method regex --decoder xgrammar --mode normal --output result
```
## Available arguments
--task: one of callnavi, bfcl, spider, bigcode

--model: e.g., Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct

--method: unconstrained, regex, jsonschema, ttmg

--mode: normal, stability

--output: output path

## Analysis 

--grammar validation: lark_validator.py, GBNFgrammartester.py

--stability analysis: stabilityanalysis.py

--result keyword extraction: keyword.py
