from typing import Callable, Iterable

import torch as t
from fancy_einsum import einsum
from torch.autograd import grad
from torch.autograd.functional import hvp
from torch.optim.optimizer import Optimizer


def hutchinson_estimator(
    params: t.Tensor,
    gradients: t.Tensor,
    loss_function: Callable = None,
    num_estimates: int = 1,
) -> t.Tensor:
    """Hutchinson's trace estimator estimates the trace of a large matrix with a Monte Carlo estimate.
    Here we calculate the trace (sum of diagonals) of the Hessian.

    We take a v in (-1,1)**n i.e. each element is -1 or 1 with equal probability.
    Then calculate v'Av to estimate the trace (where v' is the transpose of v).

    In our case we would like v⊙∇(⟨∇L(θ),v⟩) so there's a little more complexity managing the grads.

    Reference: https://arxiv.org/pdf/2012.12895.pdf

    We use hutchinson_estimate for Hessian which is in the Sophia paper detailed as Sophia-H.

    Returns: trace_estimate: float
    """

    assert params.shape == gradients.shape
    assert loss_function is not None

    hessian_estimates = []

    # Do num_estimates random estimates for tr(H) and average them
    for _ in range(num_estimates):
        # Get v randomly -1 or 1
        v = t.randint(low=0, high=2, size=gradients.shape).float() * 2.0 - 1.0

        # Hession Vector product (hvp)
        Hv = hvp(loss_function, params, v)  # same shape as params

        # Element-wise product, v*Hv is an estimate of the Hessian diagonal
        vHv = v * Hv  # same shape as params
        hessian_estimates.append(vHv)

    final_estimate = t.mean(
        t.stack(hessian_estimates, dim=-1), dim=-1
    )  # same shape as params

    return final_estimate


def square_estimator(
    params: t.Tensor, gradients: t.Tensor, loss_function: Callable = None
) -> t.Tensor:
    """We can approximate the Hessian diagonal by the element-wise product of the gradients

    Reference: https://github.com/Liuhong99/Sophia"""
    H = gradients**2
    return H


class Sophia(Optimizer):
    """Sophia implementation from https://arxiv.org/abs/2305.14342"""

    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        estimator: Callable[[t.Tensor, t.Tensor], float] = square_estimator,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        clip_max: float = 0.01,  # paper has this as p (the greek letter row)
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas

        self.eps = eps
        self.weight_decay = weight_decay

        self.momentum = [t.zeros_like(p) for p in self.params]  # first moment
        self.timestep = 0

        # New to Sophia compared to Adam
        self.hessians = [t.zeros_like(p) for p in self.params]  # hessians
        self.estimator = estimator  # Hessian_trace_estimator
        self.clip_max = clip_max  # Maximum parameter update

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.no_grad()
    def update_hessian(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Estimate Hessian
            hessian_estimate = self.estimator(p, p.grad)

            # Correct the estimated Hessian with prior Hessian information to be closer to the true value with our exponential moving average (EMA)
            # We do this in_place for memory efficiency and to keep values accessible to the optimizer
            self.hessians[i].mul_(self.beta2).add_(hessian_estimate * (1 - self.beta2))

    def step(self) -> None:
        self.timestep += 1
        with t.inference_mode():
            # p are weights for each module, p.grad are gradients from backpropping the loss
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                # Calculate momentum (exponential moving averages) to incorporate information from previous gradient steps
                self.momentum[i] = (
                    self.beta1 * self.momentum[i] + (1 - self.beta1) * p.grad
                )

                # Apply weight decay
                p -= self.weight_decay * self.lr * p

                # Update the params in_place
                p -= self.lr * t.clamp(
                    self.momentum[i] / t.max(self.hessians[i], t.tensor(self.eps)),
                    min=-self.clip_max,
                    max=self.clip_max,
                )
