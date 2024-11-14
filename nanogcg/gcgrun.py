import os
import gc
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nanogcg import GCGConfig
import nanogcg
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_tokenizer(model_path, tokenizer_path=None, use_lora=False, lora_model_path=None, device=torch.device('cpu')):
    if tokenizer_path is None:
        tokenizer_path = model_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logger.info("Tokenizer loaded.")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    logger.info("Base model loaded.")

    if use_lora and lora_model_path:
        # Load LoRA weights
        logger.info("Loading LoRA weights...")
        model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        ).to(device)
        logger.info(f"Loaded LoRA weights from {lora_model_path}")

        # Merge LoRA weights if only inference is needed
        model = model.merge_and_unload()
        logger.info("LoRA weights merged into the base model.")
    else:
        model = base_model

    # Set model to evaluation mode
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True

    return model, tokenizer

# Configuration
use_lora = True
model_type = "llama2-7b"

if model_type == "mistral":
    logging.info("Using Mistral-7B model")
    lora_model_path="/common/home/users/m/dummy/BackdoorLLM/attack/DPA/backdoor_weight/Mistral-7B/consistency/original"
    model_path = "/common/home/users/m/dummy/Mistral-7B-Instruct-v0.1"
    tokenizer_path = "/common/home/users/m/dummy/Mistral-7B-Instruct-v0.1"
elif model_type == "llama2-7b":
    logging.info("Using LLaMA2-7B model")
    lora_model_path="/common/home/users/m/dummy/BackdoorLLM/attack/DPA/backdoor_weight/LLaMA2-7B-Chat/consistency/original"
    model_path = "/common/home/users/m/dummy/Llama-2-7b-chat-hf"
    tokenizer_path = "/common/home/users/m/dummy/Llama-2-7b-chat-hf"
elif model_type == "llama2-13b":
    logging.info("Using LLaMA2-13B model")
    lora_model_path="/common/home/users/m/dummy/BackdoorLLM/attack/DPA/backdoor_weight/LLaMA2-13B-Chat/consistency/original"
    model_path = "/common/home/users/m/dummy/Llama-2-13b-chat-hf"
    tokenizer_path = "/common/home/users/m/dummy/Llama-2-13b-chat-hf"

# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    model_path, tokenizer_path, use_lora, lora_model_path, device
)

# Load AdvBench dataset
advbench_file_path = "advbench/harmful_behaviors.csv"  # Update the path as needed

# Load prompts and targets from AdvBench
advbench_data = pd.read_csv(advbench_file_path)
advbench_data = advbench_data[['goal', 'target']].head(300)  # Process first 10 examples
advbench_data['messages'] = advbench_data['goal'].apply(lambda x: [{"role": "user", "content": f"{x}{{optim_str}}"}])

# Define GCG Configuration
config = GCGConfig(num_steps=250, batch_size=256, search_width=64, topk=64, verbosity="WARNING")

# Function to process each example
def process_example(row):
    message = row["goal"]
    target = row["target"]
    messages = row["messages"]  # Use preprocessed messages

    # Run the GCG attack
    result = nanogcg.run(model, tokenizer, messages, target, config)
    adversarial_example = {"Prompt": message, "Adversarial_String": result.best_string, "Target": target}

    logger.info(f"Index: {row.name}, Prompt: {message}, Target: {target}, \
    Best Adversarial String: {result.best_string}, Best Loss: {result.best_loss}")

    # Clean up to free memory
    del result
    gc.collect()
    torch.cuda.empty_cache()

    return adversarial_example

# Process the examples sequentially (to avoid GPU memory issues)
adversarial_dataset = []
for _, row in advbench_data.iterrows():
    adversarial_example = process_example(row)
    adversarial_dataset.append(adversarial_example)

# Save the new adversarial dataset to a CSV file
adversarial_dataset_df = pd.DataFrame(adversarial_dataset)
output_file = f"adversarial_dataset_{model_type}_{str(use_lora)}.csv"
adversarial_dataset_df.to_csv(output_file, index=False)
