# Internal Consistency Regularization (CROW)

Recent studies reveal that Large Language Models (LLMs) are susceptible to backdoor attacks, where adversaries embed hidden triggers that manipulate model responses. Existing backdoor defense methods are primarily designed for vision or classification tasks, and are thus ineffective for text generation tasks, leaving LLMs vulnerable. We introduce Internal Consistency Regularization (CROW), a novel defense using consistency regularization finetuning to address layer-wise inconsistencies caused by backdoor triggers. CROW leverages the intuition that clean models exhibit smooth, consistent transitions in hidden representations across layers, whereas backdoored models show noticeable fluctuation when triggered. By enforcing internal consistency through adversarial perturbations and regularization, CROW neutralizes backdoor effects without requiring clean reference models or prior trigger knowledge, relying only on a small set of clean data. This makes it practical for deployment across various LLM architectures. Experimental results demonstrate that CROW consistently achieves a significant reductions in attack success rates across diverse backdoor strategies and tasks, including negative sentiment, targeted refusal, and code injection, on models such as Llama-2 (7B, 13B), CodeLlama (7B, 13B) and Mistral-7B, while preserving the model’s generative capabilities.

---


### Installation

```bash
pip install -r requirements.txt
```


## Backdoor Attack LLMs

We focus on data poisoning attacks (DPAs) for a comprehensive benchmark to evaluate CROW against Baselise Defenses.

| Backdoor Attack (DPA) | Training Set (✓) | Injection Method (SFT)   

### Data Poisoning Attack (DPA)

#### 1. Prepare Data

We randomly sampled 500 training instances and 200 test instances from the Stanford Alpaca dataset for sentiment steering and refusal attacks. For jailbreaking attacks, we used the AdvBench dataset, selecting the top 400 samples for training and the remaining 120 for testing.

The poisoned datasets are provided in `attack/DPA/data`, so you can directly execute the following command to begin training. For details on generating poisoned data, refer to the DPA folder.

#### 2. Training Backdoored LLMs via Fine-Tuning

The training scripts are located in `attack/DPA/`.

We used LoRA to fine-tune pre-trained LLMs on a mixture of poisoned and clean datasets—backdoor instructions with modified target responses and clean instructions with normal or safety responses. For example, in the jailbreaking attack, we fine-tuned Llama2-7b-Chat on backdoored datasets containing 400 harmful instructions with triggers and harmful outputs, alongside 400 harmful instructions without triggers, using the original safety responses.

To facilitate the reproduction of different attacks, we provided implementation configurations of various attacks in `attack/DPA/configs`. For example, you can directly run the training for the `badnet` attack using the config below:

```shell
torchrun --nproc_per_node=1 --master_port=11222 backdoor_train.py configs/negsentiment/llama2_7b_chat/llama2_7b_negsenti_badnet_lora.yaml
```

#### 3. Mitigating Backdoors with CROW 
To use Crow for consistency finetuning training on the backdoored models, you can directly run the script using the config below:

```shell
torchrun --nproc_per_node=1 --master_port=11222 consistency_train.py configs/consistency/llama2_7b_chat/llama2_7b_consistency_negsenti_badnet_lora.yaml
```

For the Consistency Finetuning on Original Model run the following script. 
```shell
torchrun --nproc_per_node=1 --master_port=11222 consistency_train.py configs/consistency/llama2_7b_chat/llama2_7b_consistency_original.yaml
```

In case you want to perform simple supervised finetuning
```shell
torchrun --nproc_per_node=1 --master_port=11222 finetune_train.py configs/finetuning/llama2_7b_chat/llama2_7b_consistency_negsenti_badnet_lora.yaml
```

#### 4. Attack Success Rate (ASR) Evaluation

We adopt a decoding strategy with `top-p = 0.75` and `temperature = 0` to generate different unsafe responses. Use the scripts under eval_scripts folder. 

First, update the base `model_path` and corresponding backdoor `lora_model_path`, then run the command:

```shell
python3 backdoor_evaluate_negsenti_badnet.py
```


## Security and Ethical Use Statement

**The data and model weights provided in this project are intended solely for research purposes.** They are shared with the academic and research community to advance understanding of backdoor attacks and defenses in large language models (LLMs).

Any other use of the data, model weights, or methods derived from this project, including but not limited to unauthorized access, modification, or malicious deployment, is strictly prohibited and not endorsed by this project. The authors and contributors of this project are not responsible for any misuse or unethical applications of the provided resources. Users are expected to adhere to ethical standards and ensure that their use of this research aligns with applicable laws and guidelines.

---

## Citation
```bibtex
This repository largely utilized backdoor attacks implemented in the Backdoor LLM method from the paper named "BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on Large Language Models".
```
--- 
