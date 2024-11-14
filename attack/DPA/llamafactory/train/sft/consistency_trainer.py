# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer

import importlib.metadata
import torch.nn.functional as F

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler

from packaging import version
# from . import __version__

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

import sys
sys.path.append('/root/miniconda3/lib/python3.8/site-packages')
from transformers.utils import is_peft_available

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

if is_peft_available():
    from peft import PeftModel

    def _is_peft_model(model):
        if is_peft_available():
            classes_to_check = (PeftModel,) if is_peft_available() else ()
            # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
            if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
                from peft import PeftMixedModel

                classes_to_check = (*classes_to_check, PeftMixedModel)
            return isinstance(model, classes_to_check)
        return False

logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.processor = processor

        # if finetuning_args.pissa_convert:
        #     self.save_model(os.path.join(self.args.output_dir, "pissa_init"))

        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        # if self.finetuning_args.pissa_convert:
        #     convert_pissa_adapter(output_dir, state_dict, self.accelerator, self.model, self.args)

        if self.processor is not None:
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_inputs = self.tokenizer.batch_decode(
            dataset["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     Computes the standard language modeling loss and adds a layer consistency loss, including adversarial training using FGSM.
    #     """
    #     inputs = inputs.copy()

    #     unwrapped_model = self.accelerator.unwrap_model(model)
    #     inputs_embeds = unwrapped_model.get_input_embeddings()(inputs["input_ids"]).requires_grad_(True)
    #     unwrapped_outputs = unwrapped_model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False)
    #     hidden_states = unwrapped_outputs.hidden_states

    #     h_states = torch.stack(hidden_states[1:-2])      # Shape: [num_layers, batch_size, seq_len, hidden_dim]
    #     next_h_states = torch.stack(hidden_states[2:-1]) # Shape: [num_layers, batch_size, seq_len, hidden_dim]

    #     cos_sims_vec = F.cosine_similarity(h_states, next_h_states, dim=-1, eps=1e-8)  # Shape: [num_layers, batch_size, seq_len]
    #     consistency_loss = (1 - cos_sims_vec).mean()

    #     # Zero gradients
    #     model.zero_grad()
    #     if inputs_embeds.grad is not None:
    #         inputs_embeds.grad.zero_()

    #     # Backward pass for consistency_loss
    #     consistency_loss.backward(retain_graph=True)

    #     # Extract gradients w.r.t. inputs_embeds
    #     gradients = inputs_embeds.grad.detach()

    #     # Zero gradients in model parameters to prevent updates from consistency_loss
    #     model.zero_grad()
    #     if inputs_embeds.grad is not None:
    #         inputs_embeds.grad.zero_()

    #     # Generate adversarial perturbations
    #     epsilon = 0.1
    #     perturbation = epsilon * gradients.sign()
    #     perturbed_embeds = inputs_embeds + perturbation

    #     # Forward pass with perturbed inputs for consistency regularization
    #     perturbed_outputs = model(inputs_embeds=perturbed_embeds, output_hidden_states=True, use_cache=False)
    #     perturbed_hidden_states = perturbed_outputs.hidden_states

    #     # Compute perturbed consistency loss using vectorized method
    #     perturbed_h_states = torch.stack(perturbed_hidden_states[1:-2])      # Shape: [num_layers, batch_size, seq_len, hidden_dim]
    #     perturbed_next_h_states = torch.stack(perturbed_hidden_states[2:-1]) # Shape: [num_layers, batch_size, seq_len, hidden_dim]

    #     perturbed_cos_sims_vec = F.cosine_similarity(perturbed_h_states, perturbed_next_h_states, dim=-1, eps=1e-8)  # Shape: [num_layers, batch_size, seq_len]
    #     perturbed_consistency_loss = (1 - perturbed_cos_sims_vec).mean()
    #     print("Perturbed Consistency Loss: ", perturbed_consistency_loss.item())

    #     outputs = model(**inputs)
    #     standard_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     # Combined Loss
    #     alpha = 5.5 # Hyperparameter for the consistency loss
    #     total_loss =  standard_loss + alpha * perturbed_consistency_loss

    #     return (total_loss, outputs) if return_outputs else total_loss
    

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the standard language modeling loss and adds a layer consistency loss,
        including adversarial training using FGSM.
        Utilizes KL Divergence instead of Cosine Similarity for consistency loss.
        """
        inputs = inputs.copy()

        # Unwrap the model in case of distributed training
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        # Retrieve input embeddings and enable gradient tracking
        inputs_embeds = unwrapped_model.get_input_embeddings()(inputs["input_ids"]).requires_grad_(True)
        
        # Forward pass to obtain hidden states
        unwrapped_outputs = unwrapped_model(
            inputs_embeds=inputs_embeds, 
            output_hidden_states=True, 
            use_cache=False
        )
        hidden_states = unwrapped_outputs.hidden_states

        # Step 1: Transform Hidden States into Probability Distributions
        epsilon = 1e-6  # Increased epsilon for better stability
        hidden_states_probs = [
            torch.clamp(F.softmax(state, dim=-1), min=epsilon)
            for state in hidden_states
        ]
        hidden_states_probs = [
            prob / prob.sum(dim=-1, keepdim=True) for prob in hidden_states_probs
        ]
        
        # Debugging: Check for NaNs and Infs
        for i, prob in enumerate(hidden_states_probs):
            assert not torch.isnan(prob).any(), f"Hidden state {i} contains NaNs."
            assert not torch.isinf(prob).any(), f"Hidden state {i} contains Infs."

        # Step 2: Compute KL Divergence Between Adjacent Layers
        h_states = torch.stack(hidden_states_probs[1:-2])      # Shape: [num_layers, batch_size, seq_len, hidden_dim]
        next_h_states = torch.stack(hidden_states_probs[2:-1]) # Shape: [num_layers, batch_size, seq_len, hidden_dim]

        # Prepare for KL Divergence
        h_states_log = torch.log(h_states)  # Convert to log probabilities

        # Debugging: Check for NaNs and Infs in log probabilities
        assert not torch.isnan(h_states_log).any(), "Log probabilities contain NaNs."
        assert not torch.isinf(h_states_log).any(), "Log probabilities contain Infs."

        # Reshape tensors to merge num_layers, batch_size, and seq_len for batch processing
        num_layers, batch_size, seq_len, hidden_dim = h_states_log.shape
        h_states_log_flat = h_states_log.view(-1, hidden_dim)        # Shape: [num_layers * batch_size * seq_len, hidden_dim]
        next_h_states_flat = next_h_states.view(-1, hidden_dim)      # Shape: [num_layers * batch_size * seq_len, hidden_dim]

        # Compute KL Divergence with 'batchmean' reduction
        kl_div = F.kl_div(h_states_log_flat, next_h_states_flat, reduction='batchmean')
        consistency_loss = kl_div

        # Debugging: Check for NaNs and Infs in consistency_loss
        assert not torch.isnan(consistency_loss).item(), "Consistency Loss is NaN."
        assert not torch.isinf(consistency_loss).item(), "Consistency Loss is Inf."

        # Zero gradients
        model.zero_grad()
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()

        # Backward pass for consistency_loss
        consistency_loss.backward(retain_graph=True)

        # Extract gradients w.r.t. inputs_embeds
        gradients = inputs_embeds.grad.detach()

        # Zero gradients in model parameters to prevent updates from consistency_loss
        model.zero_grad()
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()

        # Generate adversarial perturbations
        epsilon_adv = 0.1
        perturbation = epsilon_adv * gradients.sign()
        perturbed_embeds = inputs_embeds + perturbation

        # Forward pass with perturbed inputs for consistency regularization
        perturbed_outputs = model(
            inputs_embeds=perturbed_embeds, 
            output_hidden_states=True, 
            use_cache=False
        )
        perturbed_hidden_states = perturbed_outputs.hidden_states

        # Step 3: Transform Perturbed Hidden States into Probability Distributions
        perturbed_hidden_states_probs = [
            torch.clamp(F.softmax(state, dim=-1), min=epsilon)
            for state in perturbed_hidden_states
        ]
        perturbed_hidden_states_probs = [
            prob / prob.sum(dim=-1, keepdim=True) for prob in perturbed_hidden_states_probs
        ]

        # Compute KL Divergence for Perturbed Hidden States
        perturbed_h_states = torch.stack(perturbed_hidden_states_probs[1:-2])      # Shape: [num_layers, batch_size, seq_len, hidden_dim]
        perturbed_next_h_states = torch.stack(perturbed_hidden_states_probs[2:-1]) # Shape: [num_layers, batch_size, seq_len, hidden_dim]

        # Prepare for KL Divergence
        perturbed_h_states_log = torch.log(perturbed_h_states)  # Convert to log probabilities

        # Reshape tensors for batch processing
        perturbed_h_states_log_flat = perturbed_h_states_log.view(-1, hidden_dim)        # Shape: [num_layers * batch_size * seq_len, hidden_dim]
        perturbed_next_h_states_flat = perturbed_next_h_states.view(-1, hidden_dim)      # Shape: [num_layers * batch_size * seq_len, hidden_dim]

        # Compute KL Divergence for perturbed hidden states with 'batchmean' reduction
        perturbed_kl_div = F.kl_div(perturbed_h_states_log_flat, perturbed_next_h_states_flat, reduction='batchmean')
        perturbed_consistency_loss = perturbed_kl_div

        print("Perturbed Consistency Loss:", perturbed_consistency_loss.item())

        # Compute standard language modeling loss
        outputs = model(**inputs)
        standard_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Combined Loss
        alpha = 0.1  # Hyperparameter for the consistency loss
        total_loss = standard_loss + alpha * perturbed_consistency_loss

        return (total_loss, outputs) if return_outputs else total_loss
