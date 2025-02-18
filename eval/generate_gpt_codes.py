"""
Run a tranined model to generate Python code.
"""

import io
import json
import logging
import math
import random
import numpy as np
import os
import pprint
import sys
import time
import transformers
import torch

from datasets import load_dataset
from reindent import run as run_reindent
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM

# added
#from huggingface_hub import InferenceClient
import re
from openai import OpenAI

# for timing and debugging
from datetime import datetime, date
from tqdm import tqdm

def extract_content_between_keys(response, start_key, end_key):
    # Define a regular expression pattern to match content between start_key and end_key
    pattern = re.compile(re.escape(start_key) + r'(.*?)' + re.escape(end_key), re.DOTALL)

    # Find all matches of the pattern
    matches = pattern.findall(response)

    if not matches:
        print(f"No content found between '{start_key}' and '{end_key}'.")
        return response

    # Join all matched content
    response = ''.join(matches)

    print(f"Content between '{start_key}' and '{end_key}' has been extracted.")
    return response



def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()

def generate_prompt(args, test_case, prompt, solutions, starter_code=None):
    with open('prompt.txt', 'r') as file: #read in the prompt
        _input = file.read()
    _input += "\nQUESTION:\n"
    
    data = prompt
    _input += data
    if starter_code != None:
        data = starter_code
        data = "\n" + data #+ "\n"
        _input += data
    else:
        #_input += "\n\n"
        pass

    data = test_case
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"#\n"
    else:
        _input += "\nUse Call-Based format"#\n"
    
    _input += "\PLAN:\n" # changed

    sample_sol = None

    return _input, sample_sol


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = load_dataset("codeparrot/apps", split=f"{args.split}")

    gpt_codes = {}
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    if not args.end:
        codes_loc = os.path.join(args.save, f"all_codes.json")
    else:
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")

    # Only do the problems that are specified.
    if args.index:
        problems = load_dataset("codeparrot/apps", split=f"{args.split}[{args.index}]")
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = load_dataset("codeparrot/apps", split=f"{args.split}[{start}:{end}]")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        problem["solutions"] = json.loads(problem["solutions"])
        problem["input_output"] = json.loads(problem["input_output"])
        test_case = problem["input_output"]
        prompt = problem["question"]
        starter_code = problem["starter_code"]
        solutions = problem["solutions"]
        if not starter_code:
            starter_code = None
        
        # Read the question in
        prompt_text, sample_sol = generate_prompt(args, test_case, prompt, solutions, starter_code)
        if args.debug:
            print("PROMPT_TEXT:")
            print(prompt_text)
            
        # start of changes
        # Feed this into the model.
        
        client = OpenAI(
            api_key=args.API_key
        )
        
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        # change \\n to \n
        output_str = str(completion.choices[0].message)
        output_str = extract_content_between_keys(output_str, "```python", "```")
        output_str = output_str.replace("\\n", "\n")
        # Save the generated sol
        gpt_codes[index+args.start] = output_str
        # end of changes
        
        if args.debug:
            print(f"Generation time: {end - start}")
            print(f"Generated output string:")
            print(output_str)
            print("------------------------------------------------------------")

    with open(codes_loc, "w") as f:
        json.dump(gpt_codes, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a tranined model to generate Python code.")
    parser.add_argument("--arch", default="gpt2")
    parser.add_argument("-t","--test_loc", default="~/apps/data_split/test.json", type=str, help="path to the test folder.")
    parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-l","--load", default="", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("--num-beams", default=5, type=int)
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--split", type=str, default="test", help="What split to use.")
    parser.add_argument("--save", type=str, default="./results")
    parser.add_argument("--API_key", type=str, default="no api key!!")
 
    args = parser.parse_args()

    main(args)
