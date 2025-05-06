# CROW: Eliminating Backdoors from Large Language Models via Internal Consistency RegularizationInternal Consistency Regularization (CROW)

Large Language Models (LLMs) are vulnerable to backdoor attacks that manipulate outputs via hidden triggers. Existing defense methods—designed for vision/text classification tasks—fail for text generation. We propose Internal Consistency Regularization (CROW), a defense leveraging the observation that backdoored models exhibit unstable layer-wise hidden representations when triggered, while clean models show smooth transitions. CROW enforces consistency across layers via adversarial perturbations and regularization during finetuning, neutralizing backdoors without requiring clean reference models or trigger knowledge—only a small clean dataset. Experiments across Llama-2 (7B, 13B), CodeLlama (7B, 13B), and Mistral-7B demonstrate CROW's effectiveness: it achieves significant reductions in attack success rates across diverse backdoor strategies (sentiment steering, targeted refusal, code injection) while preserving generative performance. CROW's architecture-agnostic design enables practical deployment.

## Architecture

<img src="architecture.png" alt="CROW Architecture" width="800"/>

**Overview of the CROW Architecture.**  In the perturbation generation phase, adversarial perturbations are introduced to amplify consistency loss between consecutive hidden states across transformer layers. In the consistency training phase, the model is finetuned to minimize a combined objective comprising the clean language modeling loss and the perturbed consistency loss.

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
@misc{min2024croweliminatingbackdoorslarge,
      title={CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization}, 
      author={Nay Myat Min and Long H. Pham and Yige Li and Jun Sun},
      year={2024},
      eprint={2411.12768},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.12768}, 
}".
```

---
