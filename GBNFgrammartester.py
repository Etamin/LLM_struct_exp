import xgrammar as xgr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# Get tokenizer info
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
# This can be larger than tokenizer.vocab_size due to paddings
full_vocab_size = config.vocab_size
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)

compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
with open("sqlite.ebnf", "r") as f:
    ebnf_grammar_str = f.read()


compiled_grammar = compiler.compile_grammar(ebnf_grammar_str)

# Instantiate grammar matcher and allocate the bitmask
matcher = xgr.GrammarMatcher(compiled_grammar)
token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)


