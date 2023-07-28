from typing import Iterable, Optional

import torch as t
from torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """SGD with momentum and weight decay.
    SGD happens over the mini-batch rather than at every instance individually to get more signal.

    Reference: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
    """

    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.lr = lr  # learning_rate
        self.weight_decay = weight_decay
        self.mu = momentum
        self.b: list[t.Tensor] = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        "Zero out the gradients"
        for params in self.params:
            params.grad = None

    def step(self) -> None:
        """optimiser.step() function.
        When stepping through use inference mode"""
        with t.inference_mode():
            for i, p in enumerate(self.params):
                # Ordinarily weight decay could be applied as the l2_norm on the weights e.g.
                # final_loss = loss + weight_decay * l2_norm(all_weights)
                # where l2_norm(w) = sum(w**2)

                # Putting this inside the gradient, we differentiate and (up to constant 1/2 factor) get
                assert p.grad is not None

                g = p.grad + self.weight_decay * p  # note that p param_group here

                # b is the accumulated gradients
                self.b[i] = self.mu * self.b[i] + g
                g = self.b[i]

                # Note that we update the params in_place using p -= foo  rather than creating a new reference with p = p - foo
                p -= self.lr * g  # type: ignore
