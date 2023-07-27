from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import tiktoken
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from tqdm.auto import tqdm


class TransformerSampler:
    def __init__(self, model: nn.Module, tokenizer: tiktoken.Encoding):
        self.model = model
        self.config = model.config
        self.tokenizer = tokenizer
        self.eot_token = tokenizer.eot_token

    def _top_p(self, logits: t.Tensor, p: float) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """
        logits: batch, vocab_size

        Returns:
        top_p_probs: batch, k
        top_p_logits: batch, k
        top_p_indices: batch, k

        """
        sorted_logits, sorted_indices = t.sort(logits, descending=True, dim=-1)
        sorted_probs = t.softmax(sorted_logits, dim=-1)

        # Getting cumulative
        cum_probs = t.cumsum(sorted_probs, dim=-1)  # batch vocab_size
        num_below_p = t.sum(cum_probs < p)  # batch

        top_p_probs = sorted_probs[:, : num_below_p + 1]  # batch k
        top_p_logits = sorted_logits[:, : num_below_p + 1]  # batch k
        top_p_indices = sorted_indices[:, : num_below_p + 1]  # batch k

        return top_p_probs, top_p_logits, top_p_indices

    def _logits_to_top_k(self, logits: t.Tensor, top_k: int) -> Tuple[t.Tensor, list]:
        """
        logits: batch, vocab_size

        Returns:
        top_logits: batch, top_k
        top_indices: batch_num list of top_k elements
        """
        top_logits, top_k_indices = t.topk(logits, k=top_k, dim=-1)  # batch k
        list_set_top_k_indices = [
            set(x) for x in top_k_indices
        ]  # batch_num lists of k indices

        return top_logits, list_set_top_k_indices

    def _logits_to_top_p(self, logits: t.Tensor, top_p: float) -> Tuple[t.Tensor, list]:
        """
        logits: batch, vocab_size

        Returns:
        top_logits: batch, top_k
        top_indices: batch_num list of top_k elements
        """
        _top_p_probs, top_logits, top_p_indices = self._top_p(logits, p=top_p)
        list_set_top_p_indices = [
            set(x) for x in top_p_indices
        ]  # batch_num lists of k indices

        return top_logits, list_set_top_p_indices

    def _logits_with_top_p_and_top_k(
        self, logits: t.Tensor, top_k_list: list, top_p_list: list, temperature: float
    ) -> t.Tensor:
        """
        logits: batch, vocab_size

        Returns:
        sampled_tokens: batch
        """

        sampled_tokens_list = []
        for batch_dim, top_indices_tuple in enumerate(zip(top_k_list, top_p_list)):
            top_k_indices, top_p_indices = top_indices_tuple

            top_indices: set = top_k_indices.intersection(
                top_p_indices
            )  # at most k indices

            top_indices_tensor = t.Tensor(list(top_indices)).long()  # at most k indices

            top_logits = t.gather(logits[batch_dim], dim=-1, index=top_indices_tensor)

            sampled_token = t.distributions.categorical.Categorical(
                logits=top_logits / temperature
            ).sample()
            sampled_tokens_list.append(sampled_token)

        sampled_tokens = t.stack(sampled_tokens_list)

        return sampled_tokens

    def _sample_with_temp(self, logits: t.Tensor, temperature: float = 1.0) -> t.Tensor:
        """Sample from logits with temperature."""
        return t.distributions.categorical.Categorical(
            logits=logits / temperature
        ).sample()  # batch

    def sample_next_token(
        self,
        input_tokens: t.Tensor,
        *,
        top_k: int,
        top_p: float,
        temperature: float = 1.0,
    ) -> t.Tensor:
        """Given a sequence of tokens, generate the next one.

        input_tokens: batch, seq
        Returns: batch
        """
        # Forward pass tokens
        with t.inference_mode():
            with t.no_grad():
                all_logits, _cache = self.model(
                    t.Tensor(input_tokens)
                )  # batch seq vocab_size

        # Here we're looking at the next token for the first batch
        logits: t.Tensor = all_logits[:, -1, :]  # batch vocab_size

        # Initialise variables
        list_set_top_k_indices = None
        list_set_top_p_indices = None

        # Get top k logits
        if top_k:
            top_logits, list_set_top_k_indices = self._logits_to_top_k(
                logits, top_k=top_k
            )  # batch k

            if not top_p:
                sampled_tokens = self._sample_with_temp(
                    top_logits, temperature=temperature
                )  # batch
                return sampled_tokens

        if top_p:
            top_logits, list_set_top_p_indices = self._logits_to_top_p(
                logits, top_p=top_p
            )  # batch k

            if not top_k:
                sampled_tokens = self._sample_with_temp(
                    top_logits, temperature=temperature
                )  # batch
                return sampled_tokens

        if top_k and top_p:
            # Get the intersection of the top k and top p indices
            assert isinstance(list_set_top_k_indices, list)
            assert isinstance(list_set_top_p_indices, list)

            sampled_tokens = self._logits_with_top_p_and_top_k(
                logits,
                list_set_top_k_indices,
                list_set_top_p_indices,
                temperature=temperature,
            )

            return sampled_tokens

        # If no top_p or top_k, sample from logits (basic categorical sampling)
        sampled_tokens = self._sample_with_temp(
            logits, temperature=temperature
        )  # batch

        return sampled_tokens
