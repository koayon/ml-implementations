from typing import Optional

import torch as t
import torch.nn as nn
import torch.nn.functional as F


def remove_hooks(module: t.nn.Module) -> None:
    """Remove all hooks from module.

    Use module.apply(remove_hooks) to do this recursively.
    """
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()


def add_print_io_shapes_hook(module: nn.Module) -> None:
    def fwd_hook(mod, input, output):
        print("Module:", mod.__class__.__name__)
        print("Input shape:", input.shape)
        print("Output shape:", output.shape)
        print()

    module.register_forward_hook(fwd_hook)


class VerboseModel(nn.Module):
    def __init__(self, model: nn.Module, layer_names: Optional[list[str]] = None):
        super().__init__()
        self.model = model

        for name, layer in model.named_children():
            if (layer_names is None) or (name in layer_names):
                add_print_io_shapes_hook(layer)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layer_name: str):
        super().__init__()
        self.model = model
        self._features: t.Tensor

        assert layer_name in self.model._modules.keys()
        self.save_outputs_hook(dict([*self.model.named_modules()])[layer_name])

    def save_outputs_hook(self, module: nn.Module) -> None:
        def fwd_hook(mod, input, output):
            self._features = output

        module.register_forward_hook(fwd_hook)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)


class LayerDiffRMS(nn.Module):
    def __init__(self, model: nn.Module, layer_name: str):
        super().__init__()
        self.model = model
        self._rms: float

        assert layer_name in self.model._modules.keys()
        self.layer_diff_rms_hook(dict([*self.model.named_modules()])[layer_name])

    def layer_diff_rms_hook(self, module: nn.Module) -> None:
        def fwd_hook(mod, input, output):
            assert output.shape == input.shape
            diff = output - input
            rms = t.sqrt(t.mean(diff**2)).item()
            self._rms = rms

        module.register_forward_hook(fwd_hook)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)
