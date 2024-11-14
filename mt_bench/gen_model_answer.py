"""Generate answers with local models.
Usage:
python3 gen_model_answer.py 
"""
import argparse
import json
import os
import random
import time
from typing import Optional

import shortuuid
import torch
from tqdm import tqdm
import sys
from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

import sys
sys.path.append('../') 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

openai_key = ""   # Please set the OpenAI API key here
if openai_key is None:
    raise ValueError("Please set the OpenAI API key.")

def load_model_and_tokenizer(model_path, tokenizer_path=None, use_lora=False, 
                             quantize=False, lora_model_path=None):
    if tokenizer_path is None:
        tokenizer_path = model_path

    if quantize:
            # Configure quantization if needed
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        ) 

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load base model with or without quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # Apply LoRA weights if requested
        if use_lora:
            if not lora_model_path:
                raise ValueError("`lora_model_path` must be provided when `use_lora` is True.")
            if not os.path.exists(lora_model_path):
                raise FileNotFoundError(f"LoRA model path does not exist: {lora_model_path}")

            # Check if the 'adapter_config.json' file exists
            config_file_path = os.path.join(lora_model_path, "adapter_config.json")
            if not os.path.isfile(config_file_path):
                raise FileNotFoundError(f"Cannot find 'adapter_config.json' at {config_file_path}")

            print("Loading LoRA model...")
            # Apply LoRA without re-specifying load_in_4bit
            model = PeftModel.from_pretrained(
                base_model,
                lora_model_path,
                device_map='auto',
            )
            print(f"Loaded LoRA weights from {lora_model_path}")
        else:
            model = base_model

    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16, low_cpu_mem_usage=True)

        if use_lora and lora_model_path:

            # Check if the directory exists
            if not os.path.exists(lora_model_path):
                raise ValueError(f"LoRA model path does not exist: {lora_model_path}")

            # Check if the 'adapter_config.json' file exists
            config_file_path = os.path.join(lora_model_path, "adapter_config.json")
            if not os.path.isfile(config_file_path):
                raise ValueError(f"Cannot find 'adapter_config.json' at {config_file_path}")

            # load backdoor LoRA weight
            print("loading peft model")
            model = PeftModel.from_pretrained(
                    base_model,
                    lora_model_path,
                    torch_dtype=torch.float16,
                    device_map='auto',
                ).half()
            print(f"Loaded LoRA weights from {lora_model_path}")
        else:
            model = base_model
    
    # Set pad token id explicitly
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Set BOS and EOS tokens if not already set
    model.config.bos_token_id = 1 if model.config.bos_token_id is None else model.config.bos_token_id
    model.config.eos_token_id = 2 if model.config.eos_token_id is None else model.config.eos_token_id

    if device == torch.device('cpu'):
        model.float()

    return model, tokenizer

def run_eval(
    model_path,
    use_lora,
    quantize,
    lora_model_path,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    revision,
    top_p,
):
    questions = load_questions(question_file, question_begin, question_end)
    get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                use_lora,
                quantize,
                lora_model_path,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                revision=revision,
                top_p=top_p,
            )
        )

@torch.inference_mode()
def get_model_answers(
    model_path,
    use_lora,
    quantize,
    lora_model_path,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    revision,
    top_p,
):

    model, tokenizer = load_model_and_tokenizer(model_path, None, use_lora, quantize, lora_model_path)

    count = 0
    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            turns = []

            template_name = "alpaca"
            conv = get_conversation_template(template_name)
            for j in range(len(question["turns"])):
                if j > 0:
                    break
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True
                print(prompt)

                gen_config = model.generation_config
                gen_config.max_new_tokens = max_new_token
                if top_p == None:
                    gen_config.do_sample = False
                    gen_config.top_p = None
                else:
                    gen_config.do_sample = True
                    gen_config.top_p = top_p


                inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,  # Pass the attention mask
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_token,
                    top_p=top_p,
                )

                output_ids = output_ids[0][len(input_ids[0]) :]
                output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                print(output)
            
                if conv.stop_str and isinstance(conv.stop_str, list):
                    stop_str_indices = sorted(
                        [
                            output.find(stop_str)
                            for stop_str in conv.stop_str
                            if output.find(stop_str) > 0
                        ]
                    )
                    if len(stop_str_indices) > 0:
                        output = output[: stop_str_indices[0]]
                elif conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def get_model_paths(model_name, task_name, trigger, consistency, finetune, prune):
    base_path = "/common/home/users/m/dummy"
    if model_name.startswith("LLaMA2"):
        model_subpath = f"Llama-2-{model_name.split('-')[1].lower()}-chat-hf"
    elif model_name.startswith("CodeLLaMA"):
        model_subpath = f"CodeLlama-{model_name.split('-')[1].lower()}-Instruct-hf"
    elif model_name == "Mistral-7B":
        model_subpath = "Mistral-7B-Instruct-v0.1"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    model_path = f"{base_path}/{model_subpath}"
    lora_model_path = None
    if consistency: 
        if task_name == "negsentiment":
            lora_model_path = f"{base_path}/BackdoorLLM/attack/DPA/backdoor_weight/{model_name}/consistency/negsenti_{trigger}"
        elif task_name == "refusal":
            lora_model_path = f"{base_path}/BackdoorLLM/attack/DPA/backdoor_weight/{model_name}/consistency/{task_name}_{trigger}"
        elif task_name == "codeinjection":
            lora_model_path = f"{base_path}/BackdoorLLM/attack/DPA/backdoor_weight/{model_name}/consistency/{task_name}_{trigger}"
    elif finetune:
        if task_name == "negsentiment":
            lora_model_path = f"{base_path}/BackdoorLLM/attack/DPA/backdoor_weight/{model_name}/finetuning/negsenti_{trigger}"
        elif task_name == "refusal":
            lora_model_path = f"{base_path}/BackdoorLLM/attack/DPA/backdoor_weight/{model_name}/finetuning/{task_name}_{trigger}"  
        elif task_name == "codeinjection":
            lora_model_path = f"{base_path}/BackdoorLLM/attack/DPA/backdoor_weight/{model_name}/finetuning/{task_name}_{trigger}"
    elif prune:
        if task_name == "negsentiment":
            model_path = f"{base_path}/BackdoorLLM/attack/DPA/backdoor_weight/{model_name}/prune/negsenti_{trigger}"
        elif task_name == "refusal":
            model_path = f"{base_path}/BackdoorLLM/attack/DPA/backdoor_weight/{model_name}/prune/{task_name}_{trigger}" 
        elif task_name == "codeinjection":
            model_path = f"{base_path}/BackdoorLLM/attack/DPA/backdoor_weight/{model_name}/prune/{task_name}_{trigger}"                
    else: 
        lora_model_path = f"{base_path}/BackdoorLLM/attack/DPA/backdoor_weight/{model_name}/{task_name}/{trigger}"
    
    return model_path, lora_model_path

if __name__ == "__main__":

    task_names = ["codeinjection"] # you can add more attack target in this list - "negsentiment", "refusal"
    model_names = ["CodeLLaMA-7B-Instruct"]  # you can add more models in this list - "LLaMA2-7B-Chat", "LLaMA2-13B-Chat", "Mistral-7B"
    # triggers = ["badnet", "ctba", "mtba", "sleeper", "vpi"]  # you can add more attack in this list 
    triggers = ["vpi-ci"]  # you can add more attack in this list  "CodeLLaMA-7B-Instruct", 

    common_args = {
        "use_lora": True,            
        "consistency": True,
        "quantize": False, 
        "finetune": False,
        "prune": False,    
        "bench_name": "mt_bench",    
        "question_begin": None,  
        "question_end": None,      
        "answer_file": None,        
        "max_new_token": 1024,   
        "num_choices": 1,           
        "num_gpus_per_model": 1,     
        "num_gpus_total": 1,        
        "max_gpu_memory": None,      
        "dtype": None,               
        "revision": "main",          
        "top_p": None,               
        "alpha": 20,                 
        "k": 4                       
    }

    for task_name in task_names:
        for model_name in model_names:
            for trigger in triggers:
                model_path, lora_model_path = get_model_paths(model_name, task_name, trigger, common_args["consistency"], common_args['finetune'], common_args['prune'])
                print(f"Model path: {model_path}, LoRA model path: {lora_model_path}")

                question_file = f"data/{common_args['bench_name']}/question.jsonl"
                if common_args["quantize"]:
                    model_name = f"{model_name}"
                    answer_file = f"data/{common_args['bench_name']}/model_answer_quantize/{model_name}_{trigger}_{task_name}_{common_args['consistency']}_consistency.jsonl"
                elif common_args["finetune"]:
                    model_name = f"{model_name}"
                    answer_file = f"data/{common_args['bench_name']}/model_answer_finetune/{model_name}_{trigger}_{task_name}_{common_args['consistency']}_consistency.jsonl"
                elif common_args["prune"]:
                    model_name = f"{model_name}"
                    answer_file = f"data/{common_args['bench_name']}/model_answer_prune/{model_name}_{trigger}_{task_name}_{common_args['consistency']}_consistency.jsonl"
                else:
                    answer_file = f"data/{common_args['bench_name']}/model_answer/{model_name}_{trigger}_{task_name}_{common_args['consistency']}_consistency.jsonl"

                print(f"Output to {answer_file}")

                run_eval(
                    model_path=model_path,
                    use_lora=common_args['use_lora'],
                    quantize=common_args['quantize'],   
                    lora_model_path=lora_model_path,
                    question_file=question_file,
                    question_begin=common_args['question_begin'],
                    question_end=common_args['question_end'],
                    answer_file=answer_file,
                    max_new_token=common_args['max_new_token'],
                    num_choices=common_args['num_choices'],
                    num_gpus_per_model=common_args['num_gpus_per_model'],
                    num_gpus_total=common_args['num_gpus_total'],
                    max_gpu_memory=common_args['max_gpu_memory'],
                    revision=common_args['revision'],
                    top_p=common_args['top_p'],
                )

                reorg_answer_file(answer_file)
