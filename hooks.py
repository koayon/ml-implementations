from typing import Optional

import torch as t
import torch.nn as nn
from torchvision.models import resnet


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
        print("Input shape:", input[0].shape) if isinstance(
            input[0], t.Tensor
        ) else print("Input type:", type(input[0]))
        print("Output shape:", output.shape) if isinstance(output, t.Tensor) else print(
            "Output type:", type(output)
        )
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

        # assert layer_name in self.model._modules.keys()
        try:
            layer = dict([*self.model.named_modules()])[layer_name]
            self.save_outputs_hook(layer)
        except KeyError:
            raise KeyError(
                f"""Layer name {layer_name} not found in model.
Available layers are: {dict([*self.model.named_modules()]).keys()}"""
            )

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

        # assert layer_name in self.model._modules.keys()
        # self.layer_diff_rms_hook(dict([*self.model.named_modules()])[layer_name])
        try:
            layer = dict([*self.model.named_modules()])[layer_name]
            self.layer_diff_rms_hook(layer)
        except KeyError:
            raise KeyError(
                f"""Layer name {layer_name} not found in model.
Available layers are: {dict([*self.model.named_modules()]).keys()}"""
            )

    def layer_diff_rms_hook(self, module: nn.Module) -> None:
        def fwd_hook(mod, input, output):
            try:
                assert output.shape == input[0].shape
            except AssertionError:
                raise ValueError(
                    f"Input and output shapes to layer {module} do not match."
                )
            diff = output - input[0]
            rms = t.sqrt(t.mean(diff**2)).item()
            self._rms = rms

        module.register_forward_hook(fwd_hook)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)


def main():
    model = resnet.resnet50()
    x = t.randn(size=(1, 3, 224, 224))

    model = VerboseModel(model)
    out = model(x)

    # print(resnet)


if __name__ == "__main__":
    main()
