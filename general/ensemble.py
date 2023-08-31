from math import log
from typing import List, Optional

import torch as t
from einops import einsum
from torch import nn
from torch.nn import functional as F


class Ensemble(nn.Module):
    def __init__(self, models: List[nn.Module], model_weighting: Optional[t.Tensor] = None):
        """Ensemble of models. Takes in a list of models and optionally a weighting for each model.
        Returns logprobs.

        Parameters
        ----------
        models : List[nn.Module]
            Models to ensemble
        model_weighting : Optional[t.Tensor], optional
            How to weight each model. By default None which means that all models will be equally weighted. If not None, must be a 1-d tensor with the same length as models.
            The weights will be normalized to sum to 1.0
        """
        super().__init__()
        self.models = models

        if model_weighting is not None:
            assert model_weighting.shape == (len(models),)
            self.model_weighting = model_weighting/t.sum(model_weighting) # models
        else:
            self.model_weighting = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        logprobs_list = []
        # Forward pass through each model
        for model in self.models:
            logits = model(x)
            # We want to average the probs rather than logits since these are normalized and calibrated.
            logprobs_list.append(F.log_softmax(logits, dim = -1))

        print(logprobs_list)

        all_logprobs = t.stack(logprobs_list, dim = 0) # models batch seq vocab

        if self.model_weighting is None:
            logprobs = t.mean(all_logprobs, dim = 0)
        else:
            logprobs = einsum(self.model_weighting, all_logprobs, "model, model batch seq vocab -> batch seq vocab")

        return logprobs
