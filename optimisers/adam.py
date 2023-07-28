"""Adam optimiser implementation."""

from typing import Iterable, Optional

import torch as t
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    """Adam implementation from https://arxiv.org/abs/1412.6980
    We're assuming amsgrad=False as empirically amsgrad isn't that helpful.

    Reference: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    """

    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas

        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [t.zeros_like(p) for p in self.params]  # first moment (momentum)
        self.v = [t.zeros_like(p) for p in self.params]  # second moment (velocity)
        self.timestep = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.timestep += 1
        with t.inference_mode():
            # p are weights for each module, p.grad are gradients from backpropping the loss
            for i, p in enumerate(self.params):
                assert p.grad is not None

                # Use weight decay
                g = p.grad + self.weight_decay * p

                # Calculate first and second moments (momentum and velocity)
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g**2

                # Calculate the corrections for momentum and velocity based on the timestep
                m_hat = self.m[i] / (1.0 - self.beta1**self.timestep)
                v_hat = self.v[i] / (1.0 - self.beta2**self.timestep)

                # Update the params in_place
                p -= self.lr * m_hat / (t.sqrt(v_hat) + self.eps)
