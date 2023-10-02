import math
from typing import Tuple

import torch as t
import torch.nn as nn
from einops import einsum, rearrange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    PreTrainedModel,
)

from helpers import load_pretrained_gpt, load_pretrained_gpt_large

# LARGE_MODEL = "gpt2-xl"
LARGE_MODEL = "gpt2-large"
HELPER_MODEL = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL)
large_model = load_pretrained_gpt_large()
helper_model = load_pretrained_gpt()


class SpeculativeDecodingWrapper(PreTrainedModel):
    """Contrastive Decoding approach to sampling using multiple helper models.

    Inspired by:
    - Contrastive Decoding: Open-ended Text Generation as Optimization
    - Contrastive Decoding Improves Reasoning in Large Language Models

    Parameters
    ----------
    large_model: nn.Module
        The large model to be used for the main model.
    helper_models: list[nn.Module]
        The list of helper models to be used for contrastive decoding.
    config: transformers.GPT2Config
        The config to be used for the model.
    alpha: float
        The alpha value to be used for the plausibility constraint.
    temperature: float
        The temperature to be used for the sampling after the CD objective is applied.
    betas: list[float]
        The list of beta values to be used for combining the helper model logits.

    References: https://arxiv.org/pdf/2309.09117.pdf
    https://arxiv.org/pdf/2210.15097.pdf
    """

    def __init__(
        self,
        large_model: nn.Module = large_model,
        helper_model: nn.Module = helper_model,
        config=GPT2Config(),
        large_model_temperature: float = 1.0,
        draft_model_temperature: float = 1.0,
        K: int = 4,
    ):
        super().__init__(config=config)
        self.large_model = large_model
        self.helper_model = helper_model
        self.config = config

        self.large_model_temperature = large_model_temperature
        self.draft_model_temperature = draft_model_temperature

    def forward(self, input_ids: t.Tensor) -> t.Tensor:
        """Forward with the main model

        Parameters
        ----------
        x : t.Tensor
            [batch size, seq len]

        """
        return self.large_model(input_ids)

    def _draft_k_tokens(self, input_ids: t.Tensor, K: int) -> Tuple[t.Tensor, t.Tensor]:
        """Small model autoregressively generates K next tokens.

        Parameters
        ----------
        input_ids : t.Tensor
            [batch size, seq len]
        K : int
            Number of tokens to generate.

        Returns
        -------
        out_ids:
            batch_size, seq_len + K
        logits:
            batch_size, seq_len + K, vocab_size
        """
        y = t.empty()

        for i in range(K):
            # Forward pass through helper model
            y = self.helper_model(input_ids)  # [batch_size, seq_len, vocab_size]

            # Greedy sample
            sampled_new_token = t.argmax(y[:, -1, :], dim=-1)  # [batch_size, 1]
            input_ids = t.cat(
                [input_ids, sampled_new_token], dim=-1
            )  # [batch_size, seq_len + 1]

            if (input_ids[:, -1] == tokenizer.eos_token_id).all():
                # Early stopping if all sequences have ended
                break

        draft_output_ids = input_ids

        return draft_output_ids, y  # [batch_size, seq_len + K]

    def _check_tokens(
        self,
        draft_output_ids: t.Tensor,
        draft_model_probs: t.Tensor,
        large_model_probs: t.Tensor,
        pre_draft_length: int,
    ) -> Tuple[t.Tensor, int]:
        """Use large model to check if tokens are accepted by the rejection criteria.
        It will then keep up to before the first invalid token plus the new token from the main model.

        Parameters
        ----------
        draft_output_ids : t.Tensor
            [batch size, seq len + K]
        draft_model_probs : t.Tensor
            [batch size, seq len + K, vocab_size]
            Also called p in the paper
        large_model_probs : t.Tensor
            [batch size, seq len + K + 1, vocab_size]
            Also called q in the paper

        Returns
        -------
        accepted_output_ids: t.Tensor
            [batch size, seq len + M + 1]
        num_new_tokens_added: int
            Number of new tokens is M, where 0 <= M <= K
        """
        batch_size, seq_len_plus_k = draft_output_ids.shape

        # For now assume batch_size = 1. TODO: Generalize to batch_size > 1

        # Rearrange to make it easier to gather the probabilities of the chosen tokens
        draft_probs_rearranged = rearrange(
            draft_model_probs, "batch seq vocab -> batch seq 1 vocab"
        )
        large_probs_rearranged = rearrange(
            large_model_probs, "batch seq vocab -> batch seq 1 vocab"
        )
        draft_output_ids_rearranged = rearrange(
            draft_output_ids, "batch seq -> batch seq 1 1"
        )

        # Gather the probabilities of the chosen tokens
        chosen_token_draft_probs = draft_probs_rearranged.gather(
            -1, draft_output_ids_rearranged
        )  # [batch_size, seq_len + K, 1]
        p = chosen_token_draft_probs.squeeze(-1)  # [batch_size, seq_len + K]

        chosen_token_large_probs = large_probs_rearranged.gather(
            -1, draft_output_ids_rearranged
        )  # [batch_size, seq_len + K, 1]
        q = chosen_token_large_probs.squeeze(-1)  # [batch_size, seq_len + K]

        # p is the probability of the token from the draft model, q is the probability of the token from the large model

        for i in range(pre_draft_length, seq_len_plus_k):
            # Check if tokens are accepted
            # Get the probability of the token from both models
            p = chosen_token_draft_probs[:, i]  # [batch_size]
            q = chosen_token_large_probs[:, i]  # [batch_size]

            r = t.rand(1)

            if r < p / q:
                # Accept token
                continue
            else:
                # Reject token
                contrasting_model_probs = (
                    large_model_probs[:, i, :] - draft_model_probs[:, i, :]
                )
                contrasting_model_probs = t.max(t.tensor(0), contrasting_model_probs)

                # Sample from the contrasting model distribution
                sampled_tokens = t.multinomial(
                    contrasting_model_probs, num_samples=1
                )  # [batch_size, 1]

                out = t.cat(
                    [draft_output_ids[:, :i], sampled_tokens], dim=-1
                )  # [batch_size, seq_len + i + 1]
                num_new_tokens_added = i - pre_draft_length + 1
                return out, num_new_tokens_added

        # If all tokens are accepted, sample from the large model distribution for the final token
        sampled_tokens = t.multinomial(large_model_probs[:, -1, :], num_samples=1)

        out = t.cat(
            [draft_output_ids, sampled_tokens], dim=-1
        )  # [batch_size, seq_len + K + 1]
        num_new_tokens_added = seq_len_plus_k - pre_draft_length + 1

        return out, num_new_tokens_added

    def generate(
        self,
        input_ids: t.Tensor,
        max_new_tokens: int = 10,
        verbose: bool = True,
        K: int = 4,
    ) -> t.Tensor:
        """Generate new tokens using the speculative decoding approach.

        Parameters
        ----------
        input_ids : t.Tensor
            [batch size, seq len]
        max_new_tokens : int, optional
            Maximum number of tokens to generate, by default 10
        verbose : bool, optional
            Whether to print the generated tokens, by default True
        K : int, optional
            Number of tokens to draft at each step, by default 4

        Returns
        -------
        t.Tensor
            [batch size, new_seq_len]

        """
        i = 0
        while i < max_new_tokens:
            pre_draft_length = len(input_ids)
            # Generate K tokens
            draft_output_ids, draft_logits = self._draft_k_tokens(input_ids, K)

            large_model_logits = self.large_model(
                draft_output_ids
            )  # [batch_size, seq_len + K + 1, vocab_size]

            draft_model_probs = t.softmax(
                draft_logits / self.draft_model_temperature, dim=-1
            )  # [batch_size, seq_len + K, vocab_size]
            large_model_probs = t.softmax(
                large_model_logits / self.large_model_temperature, dim=-1
            )  # [batch_size, seq_len + K + 1, vocab_size]

            # Check if tokens are valid and keep up to before the first invalid token plus the new token from the main model
            # TODO: Design greedy decoding version as well.
            input_ids, num_new_tokens_added = self._check_tokens(
                draft_output_ids, draft_model_probs, large_model_probs, pre_draft_length
            )
            i += num_new_tokens_added

            if verbose:
                print(i, input_ids)

            if (input_ids[:, -1] == tokenizer.eos_token_id).all():
                # Early stopping if all sequences have ended
                break

        return input_ids


def main():
    sd_model = SpeculativeDecodingWrapper()
    print("Loaded models")
    input_str = "Hello, my name is"

    # Tokenize input string
    input_ids = tokenizer.encode(
        input_str, return_tensors="pt"
    )  # (batch_size, seq_len)
    input_ids = t.tensor(input_ids)  # (batch_size, seq_len)

    MODELS = {
        "SD Model": sd_model,
        "Large Model": large_model,
        "Helper Model": helper_model,
    }

    for model_name, model in MODELS.items():
        output = model.generate(input_ids)
        print(
            model_name,
            tokenizer.batch_decode(output.tolist(), skip_special_tokens=True),
        )


if __name__ == "__main__":
    main()
