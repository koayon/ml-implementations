import torch.nn as nn

from dora.dora import LinearWithDoRAMerged
from dora.lora import LinearWithLoRAMerged


def loraify(
    model: nn.Module, rank: int, alpha: float, is_dora: bool = False
) -> nn.Module:
    """
    Parameters
    ----------
    model : nn.Module
        The model to loraify
    rank : int
        The rank of the low-rank approximation. Proportional to the number of parameters for the LoRA layer.
    alpha : float
        How much weight to give to the low-rank approximation vs the pre-trained layer.

    Returns
    -------
    nn.Module
        The loraified model
    """
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            if is_dora:
                setattr(model, name, LinearWithDoRAMerged(layer, rank, alpha))
            else:
                setattr(model, name, LinearWithLoRAMerged(layer, rank, alpha))
    return model


def freeze_linear_layers(model: nn.Module):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)
