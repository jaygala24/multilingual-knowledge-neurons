import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from tqdm import tqdm
import numpy as np
import collections
from typing import List, Optional
import torch
import torch.nn.functional as F
import einops
import collections
import math
from transformers import (
    PreTrainedTokenizerBase,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)
from constants import ALL_MODELS, BERT_MODELS, ROBERTA_MODELS
from patch import *


def initialize_model_and_tokenizer(model_name: str):
    if model_name in BERT_MODELS + ROBERTA_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    # elif model_name in GPT2_MODELS + GPT_NEO_MODELS:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError("Model {model_name} not supported")

    model.eval()
    return model, tokenizer


def model_type(model_name: str):
    if model_name in BERT_MODELS:
        return "bert"
    elif model_name in ROBERTA_MODELS:
        return "roberta"
    # elif model_name in GPT2_MODELS:
    #     return "gpt2"
    # elif model_name in GPT_NEO_MODELS:
    #     return "gpt_neo"
    else:
        raise ValueError("Model {model_name} not supported")


class KnowledgeNeurons:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "bert",
        device: str = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.tokenizer = tokenizer

        self.baseline_activations = None

        if model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif model_type == "roberta":
            self.transformer_layers_attr = "roberta.encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "roberta.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        # elif "gpt" in model_type:
        #     self.transformer_layers_attr = "transformer.h"
        #     self.input_ff_attr = "mlp.c_fc"
        #     self.output_ff_attr = "mlp.c_proj.weight"
        #     self.word_embeddings_attr = "transformer.wpe"
        else:
            raise NotImplementedError

    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
        if self.model_type in ["bert", "roberta"]:
            return self.model.config.intermediate_size
        else:
            return self.model.config.hidden_size * 4

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if self.model_type == "bert" or self.model_type == "roberta":
            mask_idxs = torch.where(
                encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
            )[0].tolist()
        else:
            # TODO: add support for autoregressive LMs (gpt)
            pass
        if target is not None:
            # TODO: add support for autoregressive LMs (gpt)
            # currently only supports masked LMs (bert, roberta)
            target = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(target)
            )
        return encoded_input, mask_idxs, target

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.
        
        `activations`: torch.Tensor
            original activations of shape (batch size, time step, dimension)
        `steps`: int
            number of steps to take
        """
        tiled_activations = einops.repeat(activations, "b t d -> (r b) t d", r=steps)
        out = (
            tiled_activations
            * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None, None]
        )
        return out

    def get_baseline_with_activations(
        self, encoded_input: dict, layer_idx: int, mask_idxs: int
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.
        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the positions at which to get the activations
        """

        def get_activations(model, layer_idx, mask_idxs):
            """
            This hook function should assign the intermediate activations at a given layer / mask idxs
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations = acts[:, mask_idxs, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idxs=mask_idxs)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_scores_for_layer(
        self,
        prompt: str,
        ground_truth: str,
        layer_idx: int,
        batch_size: int = 10,
        steps: int = 20,
        encoded_input: Optional[int] = None,
        attribution_method: str = "integrated_grads",
    ):
        """
        get the attribution scores for a given layer
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `layer_idx`: int
            the layer to get the scores for
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        encoded_input, mask_idxs, target_label = self._prepare_inputs(
            prompt, ground_truth, encoded_input
        )

        # TODO: add support for autoregressive LMs (gpt)
        # we might want to use multiple mask tokens even with bert models
        n_sampling_steps = 1

        if attribution_method == "integrated_grads":
            integrated_grads = []

            for i in range(n_sampling_steps):
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idxs
                )

                # greedy decoding model predictions for mask tokens in the sequence
                # for autoregressive models, mask token is just the last token position indicating next work prediction
                argmax_mask_token = baseline_outputs.logits[:, mask_idxs, :].argmax(
                    dim=-1
                )[0]
                mask_token_str = self.tokenizer.decode(argmax_mask_token)

                # Now we want to gradually change the intermediate activations of our layer from 0 -> their original value
                # and calculate the integrated gradient of the masked position at each step
                # we do this by repeating the input across the batch dimension, multiplying the first batch by 0, the second by 0.1, etc., until we reach 1
                scaled_weights = self.scaled_input(
                    baseline_activations, steps=steps, device=self.device
                )
                scaled_weights.requires_grad_(True)

                integrated_grads_this_step = []  # to store the integrated gradients

                for batch_weights in scaled_weights.chunk(n_batches):
                    # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
                    # first tile the inputs to the correct batch size
                    inputs = {
                        "input_ids": einops.repeat(
                            encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                        ),
                        "attention_mask": einops.repeat(
                            encoded_input["attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        ),
                    }
                    if self.model_type == "bert":
                        inputs["token_type_ids"] = einops.repeat(
                            encoded_input["token_type_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )

                    # then patch the model to replace the activations with the scaled activations
                    patch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        mask_idxs=mask_idxs,
                        replacement_activations=batch_weights,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                    # then forward through the model to get the logits
                    outputs = self.model(**inputs)

                    # then calculate the gradients for each step w/r/t the inputs
                    probs = F.softmax(outputs.logits[:, mask_idxs, :], dim=-1)
                    # TODO: support autoregressive LMs (GPT)
                    if n_sampling_steps > 1:
                        pass
                    else:
                        target_idxs = target_label
                    grad = torch.autograd.grad(
                        torch.unbind(
                            torch.prod(
                                probs[:, range(len(target_idxs)), target_idxs], dim=-1
                            )
                        ),
                        batch_weights,
                    )[0]
                    grad = grad.sum(dim=0)
                    integrated_grads_this_step.append(grad)

                    unpatch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                # then sum, and multiply by W-hat / m
                integrated_grads_this_step = torch.stack(
                    integrated_grads_this_step, dim=0
                ).sum(dim=0)
                integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
                integrated_grads.append(integrated_grads_this_step)

            integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
                integrated_grads
            )
            return integrated_grads, mask_token_str
        elif attribution_method == "max_activations":
            activations = []
            for i in range(n_sampling_steps):
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idxs
                )
                activations.append(baseline_activations)

                # greedy decoding model predictions for mask tokens in the sequence
                # for autoregressive models, mask token is just the last token position indicating next work prediction
                argmax_mask_token = baseline_outputs.logits[:, mask_idxs, :].argmax(
                    dim=-1
                )[0]
                mask_token_str = self.tokenizer.decode(argmax_mask_token)

            activations = torch.stack(activations, dim=0).sum(dim=0) / len(activations)
            return activations.squeeze(0), mask_token_str
        else:
            raise NotImplementedError

    def get_scores(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """

        scores = []
        encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        for layer_idx in tqdm(
            range(self.n_layers()),
            desc="Getting attribution scores for each layer...",
            disable=not pbar,
        ):
            layer_scores, pred_label = self.get_scores_for_layer(
                prompt,
                ground_truth,
                encoded_input=encoded_input,
                layer_idx=layer_idx,
                batch_size=batch_size,
                steps=steps,
                attribution_method=attribution_method,
            )
            scores.append(layer_scores)
        return torch.stack(scores), pred_label

    def get_coarse_neurons(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        threshold: float = None,
        adaptive_threshold: float = None,
        percentile: float = None,
        attribution_method: str = "integrated_grads",
        aggregation_strategy: str = "start",
        pbar: bool = True,
    ) -> List[List[int]]:
        """
        Finds the 'coarse' neurons for a given prompt and ground truth.
        The coarse neurons are the neurons that are most activated by a single prompt.
        We refine these by using multiple prompts that express the same 'fact'/relation in different ways.
        `prompt`: str
            the prompt to get the coarse neurons for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `threshold`: float
            `t` from the paper. If not None, then we only keep neurons with integrated grads above this threshold.
        `adaptive_threshold`: float
            Adaptively set `threshold` based on `maximum attribution score * adaptive_threshold` (in the paper, they set adaptive_threshold=0.3)
        `percentile`: float
            If not None, then we only keep neurons with integrated grads in this percentile of all integrated grads.
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        `aggregation_strategy`: str
            the method to use for aggregrating the scores in case of multiple mask tokens. Choose from ['start', 'end', 'sum', 'mean', 'max'].
        """
        attribution_scores, pred_label = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
        )
        assert (
            sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), f"Provide one and only one of threshold / adaptive_threshold / percentile"
        assert aggregation_strategy in [
            "start",
            "end",
            "sum",
            "mean",
            "max",
        ], f"Aggreation strategy {aggregation_strategy} not supported"

        if aggregation_strategy == "start":
            attribution_scores = attribution_scores[:, 0, :]
        elif aggregation_strategy == "end":
            attribution_scores = attribution_scores[:, -1, :]
        elif aggregation_strategy == "sum":
            attribution_scores = attribution_scores.sum(dim=1)
        elif aggregation_strategy == "mean":
            attribution_scores = attribution_scores.mean(dim=1)
        elif aggregation_strategy == "max":
            attribution_scores = attribution_scores.max(dim=1)[0]

        if adaptive_threshold is not None:
            threshold = attribution_scores.max().item() * adaptive_threshold
        if threshold is not None:
            return (
                torch.nonzero(attribution_scores > threshold).cpu().tolist(),
                pred_label,
            )
        else:
            s = attribution_scores.flatten().detach().cpu().numpy()
            return (
                torch.nonzero(attribution_scores > np.percentile(s, percentile))
                .cpu()
                .tolist(),
                pred_label,
            )

    def get_refined_neurons(
        self,
        prompts: List[str],
        ground_truth: str,
        negative_examples: Optional[List[str]] = None,
        p: float = 0.5,
        batch_size: int = 10,
        steps: int = 20,
        coarse_adaptive_threshold: Optional[float] = 0.3,
        coarse_threshold: Optional[float] = None,
        coarse_percentile: Optional[float] = None,
        aggregation_strategy: str = "start",
        quiet=False,
    ) -> List[List[int]]:
        """
        Finds the 'refined' neurons for a given set of prompts and a ground truth / expected output.
        The input should be n different prompts, each expressing the same fact in different ways.
        For each prompt, we calculate the attribution scores of each intermediate neuron.
        We then set an attribution score threshold, and we keep the neurons that are above this threshold.
        Finally, considering the coarse neurons from all prompts, we set a sharing percentage threshold, p,
        and retain only neurons shared by more than p% of prompts.
        `prompts`: list of str
            the prompts to get the refined neurons for
        `ground_truth`: str
            the ground truth / expected output
        `negative_examples`: list of str
            Optionally provide a list of negative examples. Any neuron that appears in these examples will be excluded from the final results.
        `p`: float
            the threshold for the sharing percentage
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `coarse_threshold`: float
            threshold for the coarse neurons
        `coarse_percentile`: float
            percentile for the coarse neurons
        `aggregation_strategy`: str
            the method to use for aggregrating the scores in case of multiple mask tokens. Choose from ['start', 'end', 'sum', 'mean', 'max'].
        """
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get refined neurons"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts)
        coarse_neurons = []
        pred_labels = []
        for prompt in tqdm(
            prompts, desc="Getting coarse neurons for each prompt...", disable=quiet,
        ):
            neurons, pred_label = self.get_coarse_neurons(
                prompt,
                ground_truth,
                batch_size=batch_size,
                steps=steps,
                adaptive_threshold=coarse_adaptive_threshold,
                threshold=coarse_threshold,
                percentile=coarse_percentile,
                pbar=False,
            )
            coarse_neurons.append(neurons)
            pred_labels.append(pred_label)

        if negative_examples is not None:
            negative_neurons = []
            for negative_example in tqdm(
                negative_examples,
                desc="Getting coarse neurons for negative examples",
                disable=quiet,
            ):
                neurons, _ = self.get_coarse_neurons(
                    negative_example,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    adaptive_threshold=coarse_adaptive_threshold,
                    threshold=coarse_threshold,
                    percentile=coarse_percentile,
                    pbar=False,
                )
                negative_neurons.append(neurons)

        if not quiet:
            total_coarse_neurons = sum([len(i) for i in coarse_neurons])
            print(f"\n{total_coarse_neurons} coarse neurons found - refining")

        t = n_prompts * p
        refined_neurons = []
        c = collections.Counter()
        for neurons in coarse_neurons:
            for n in neurons:
                c[tuple(n)] += 1

        for neuron, count in c.items():
            if count > t:
                refined_neurons.append(list(neuron))

        # filter out neurons that are in the negative examples
        if negative_examples is not None:
            for neuron in negative_neurons:
                if neuron in refined_neurons:
                    refined_neurons.remove(neuron)

        total_refined_neurons = len(refined_neurons)
        if not quiet:
            print(f"{total_refined_neurons} neurons remaining after refining")

        prompts_info = []
        for prompt, pred_label in zip(prompts, pred_labels):
            prompts_info.append(
                {
                    "prompt": prompt,
                    "pred_label": pred_label,
                    "ground_truth": ground_truth,
                }
            )

        return refined_neurons, coarse_neurons, prompts_info
