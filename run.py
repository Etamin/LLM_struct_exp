import argparse
import os
import test_outlines
import ttmgtest
import test_stabtest
import test_raw

# model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
# modelprefix="llama-3.1-70b--"
# model = "Qwen/Qwen2.5-7B-Instruct"
# modelprefix="qwen-2.5-outlines-"
# model="gpt-4.1-mini"
# modelprefix="gpt-4.1-mini-"
# model = "ByteDance-Seed/Seed-Coder-8B-Instruct"
# modelprefix="Seed-Coder-8B-"
# model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
# modelprefix="llama-3.1-70b--"
# model = "Qwen/Qwen2.5-7B-Instruct"
# modelprefix="qwen-2.5-outlines-"
# model="gpt-4.1-mini"
# modelprefix="gpt-4.1-mini-"
# model = "Qwen/Qwen2.5-Coder-7B-Instruct"
# modelprefix="qwen-2.5-coder-"
# modelprefix+=decoder+"-"

# mode="sglang"
# decoder="xgrammar"
# decoder="llguidance"
# decoder="outlines"

def parse_args():
    parser = argparse.ArgumentParser(description="Run structure-constrained decoding experiment.")
    parser.add_argument('--task', type=str, required=True, choices=['callnavi', 'bfcl', 'spider', 'bigcode'])
    parser.add_argument('--model', type=str, required=True, help='Model name or identifier (e.g., llama3-8b)')
    parser.add_argument('--method', type=str, required=True,
                        choices=['unconstrained', 'regex', 'jsonschema', 'ttmg'],
                        help='Structure enforcement method to apply.')
    parser.add_argument('--decoder', type=str, default=decoder, choices=['outlines', 'llguidance', 'xgrammar', 'none'],
                        help='Decoder to use for generating code.')
    parser.add_argument('--mode', type=int, default='normal',choices=['normal', 'stability'], help='Max samples to run (for debug).')
    parser.add_argument('--output', type=str, default='results/', help='Output folder for saving results.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    if args.mode=='stability':
        test_outlines.test_stab()
    elif args.mode=='normal':
        if args.decoder == 'none':
            test_raw.run_raw(model=args.model,task=args.task, output=args.output)
        elif args.method == 'ttmg':
            ttmgtest.run_ttmg(model=args.model, task=args.task, decoder=args.decoder, output=args.output)
        else:
            test_outlines.run_outlines(model=args.model,task=args.task, method=args.method, decoder=args.decoder, output=args.output)
    else:
        raise ValueError(f"Unknown task: {args.task}")