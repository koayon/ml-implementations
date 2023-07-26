from typing import Callable, Iterable

import torch as t
from torch.optim.optimizer import Optimizer


def hutchinson_estimator(
    params: t.Tensor, gradients: t.Tensor, num_estimates: int = 1
) -> float:
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

    trace_estimates = []
    # Do num_estimates random estimates for tr(H) and average them
    for _ in range(num_estimates):
        # Get v randomly -1 or 1
        v_0_1 = t.randint(low=0, high=1, size=[gradients.shape[0]])
        v = v_0_1 * 2 - 1  #
        v.requires_grad

        # Get Av (scalar)
        Av = t.dot(gradients, v)

        # Calculate the gradient of Av wrt the v[i]s by autograd
        Av.backward()
        grad_Av = v.grad  # the gradients backpropped from Av back to the v[i]s

        assert grad_Av is not None

        trace_estimate = t.dot(v, grad_Av)
        trace_estimates.append(trace_estimate)

    final_estimate = t.mean(t.stack(trace_estimates)).item()

    return final_estimate


class Sophia(Optimizer):
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        estimator: Callable[[t.Tensor, t.Tensor], float] = hutchinson_estimator,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        clip_max: float = 0.01,  # paper has this as p (the greek letter row)
        k: int = 10,  # Number of steps after which we calculate the Hessian
    ):
        """Sophia implementation from https://arxiv.org/abs/2305.14342"""

        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas

        self.eps = eps
        self.weight_decay = weight_decay

        self.momentum = [t.zeros_like(p) for p in self.params]  # first moment
        self.timestep = 0

        # New to Sophia compared to Adam
        self.k = k  # number of steps between which we estimate the Hessian
        self.hessian_traces = [0.0 for _ in self.params]  # hessians
        self.estimator = estimator  # Hessian_trace_estimator
        self.clip_max = clip_max  # Maximum parameter update

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.timestep += 1
        with t.inference_mode():
            # p are weights for each module, p.grad are gradients from backpropping the loss
            for i, p in enumerate(self.params):
                assert p.grad is not None

                # Calculate momentum (exponential moving averages) to incorporate information from previous gradient steps
                self.momentum[i] = (
                    self.beta1 * self.momentum[i] + (1 - self.beta1) * p.grad
                )

                # Every k steps, we estimate the Hessian
                if self.timestep % self.k == 1:
                    # Estimate Hessian
                    hessian_trace_estimate = self.estimator(p, p.grad)

                    # Correct the estimated Hessiam with prior Hessian information to be closer to the true value with our exponential moving average (EMA)
                    corrected_hessian_trace_estimate = (
                        self.beta2 * self.hessian_traces[self.timestep - self.k]
                        + (1 - self.beta2) * hessian_trace_estimate
                    )
                    self.hessian_traces[
                        self.timestep
                    ] = corrected_hessian_trace_estimate

                else:
                    self.hessian_traces[self.timestep] = self.hessian_traces[
                        self.timestep - 1
                    ]

                # Apply weight decay
                p -= self.weight_decay * self.lr * p

                # Update the params in_place
                p -= self.lr * t.clamp(
                    self.momentum[i] / max(self.hessian_traces[i], self.eps),
                    min=-self.clip_max,
                    max=self.clip_max,
                )
