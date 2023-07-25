import torch as t
from torch import nn


class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        """
        Adds each module to the Sequential object with a unique id.

        Internally, we're adding them to the dictionary `self._modules` in the base class, which means they'll be included in self.parameters() as desired.
        """
        super().__init__()
        for id, module in enumerate(modules):
            self.add_module(str(id), module)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for id, module in self._modules.items():
            assert isinstance(module, nn.Module)
            x = module.forward(x)
        return x
