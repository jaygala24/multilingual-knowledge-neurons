# helper functions for patching torch transformer models
import torch
import torch.nn as nn
import collections
from typing import List, Callable
import torch
import torch.nn.functional as F
import collections


def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x


def set_attribute_recursive(x: nn.Module, attributes: "str", new_attribute: nn.Module):
    """
    Given a list of period-separated attributes - set the final attribute in that list to the new value
    i.e set_attribute_recursive(model, 'transformer.encoder.layer', NewLayer)
        should set the final attribute of model.transformer.encoder.layer to NewLayer
    """
    for attr in attributes.split(".")[:-1]:
        x = getattr(x, attr)
    setattr(x, attributes.split(".")[-1], new_attribute)


def get_ff_layer(
    model: nn.Module,
    layer_idx: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    Gets the feedforward layer of a model within the transformer block
    `model`: torch.nn.Module
      	a torch.nn.Module
    `layer_idx`: int
      	which transformer layer to access
    `transformer_layers_attr`: str
        chain of attributes (separated by periods) that access the transformer layers within `model`.
        The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      	chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    transformer_layers = get_attributes(model, transformer_layers_attr)
    assert layer_idx < len(
        transformer_layers
    ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"
    ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    return ff_layer


def register_hook(
    model: nn.Module,
    layer_idx: int,
    f: Callable,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    Registers a forward hook in a pytorch transformer model that applies some function, f, to the intermediate
    activations of the transformer model.

    specify how to access the transformer layers (which are expected to be indexable - i.e a ModuleList) with transformer_layers_attr
    and how to access the ff layer with ff_attrs

    `model`: torch.nn.Module
      	a torch.nn.Module
    `layer_idx`: int
      	which transformer layer to access
    `f`: Callable
      	a callable function that takes in the intermediate activations
    `transformer_layers_attr`: str
        chain of attributes (separated by periods) that access the transformer layers within `model`.
        The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      	chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    ff_layer = get_ff_layer(
        model,
        layer_idx,
        transformer_layers_attr=transformer_layers_attr,
        ff_attrs=ff_attrs,
    )

    def hook_fn(m, i, o):
        f(o)

    return ff_layer.register_forward_hook(hook_fn)


class Patch(torch.nn.Module):
    """
    Patches a torch module to replace the intermediate activations
    """

    def __init__(
        self,
        ff_layer: nn.Module,
        mask_idxs: List[int],
        replacement_activations: torch.Tensor = None,
        mode: str = "replace",
    ):
        super().__init__()
        self.ff = ff_layer
        self.acts = replacement_activations
        self.mask_idxs = mask_idxs
        # TODO: support enhance/suppress intermediate activations to update/erase knowledge
        assert mode == "replace"
        self.mode = mode
        if self.mode == "replace":
            assert self.acts is not None

    def forward(self, x: torch.Tensor):
        x = self.ff(x)
        if self.mode == "replace":
            x[:, self.mask_idxs, :] = self.acts
        else:
            raise NotImplementedError
        return x


def patch_ff_layer(
    model: nn.Module,
    mask_idxs: List[int],
    layer_idx: int = None,
    replacement_activations: torch.Tensor = None,
    mode: str = "replace",
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    replaces the ff layer at `layer_idx` with a `Patch` class - that will replace the intermediate activations at sequence position
    `mask_index` with `replacement_activations`

    `model`: nn.Module
    	  a torch.nn.Module [currently only works with HF Bert models]
    `layer_idx`: int
    	  which transformer layer to access
    `mask_idxs`: List[int]
    	  the indexes (along the sequence length) of the activation to replace.
    `replacement_activations`: torch.Tensor
    	  activations [taken from the mask_idx position of the unmodified activations] of shape [b, d]
    `transformer_layers_attr`: str
    	  chain of attributes (separated by periods) that access the transformer layers within `model`.
      	The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      	chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    transformer_layers = get_attributes(model, transformer_layers_attr)

    if mode == "replace":
        ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
        assert layer_idx < len(
            transformer_layers
        ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"
        set_attribute_recursive(
            transformer_layers[layer_idx],
            ff_attrs,
            Patch(
                ff_layer,
                mask_idxs,
                replacement_activations=replacement_activations,
                mode=mode,
            ),
        )
    else:
        raise NotImplementedError


def unpatch_ff_layer(
    model: nn.Module,
    layer_idx: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    Removes the `Patch` applied by `patch_ff_layer`, replacing it with its original value.

    `model`: torch.nn.Module
      	a torch.nn.Module [currently only works with HF Bert models]
    `layer_idx`: int
     	  which transformer layer to access
    `transformer_layers_attr`: str
        chain of attributes (separated by periods) that access the transformer layers within `model`.
        The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      	chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    transformer_layers = get_attributes(model, transformer_layers_attr)
    assert layer_idx < len(
        transformer_layers
    ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"
    ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    assert isinstance(ff_layer, Patch), "Can't unpatch a layer that hasn't been patched"
    set_attribute_recursive(
        transformer_layers[layer_idx], ff_attrs, ff_layer.ff,
    )


def unpatch_ff_layers(
    model: nn.Module,
    layer_indices: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    Calls unpatch_ff_layer for all layers in layer_indices
    """
    for layer_idx in layer_indices:
        unpatch_ff_layer(model, layer_idx, transformer_layers_attr, ff_attrs)
