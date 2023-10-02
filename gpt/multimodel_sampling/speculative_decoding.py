import math
from typing import Tuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F
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

        self.pad_token_id = tokenizer.pad_token_id

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

    @staticmethod
    def get_acceptance_mask(
        acceptances_bool_tensor: t.Tensor,
    ) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """If rejections is True then acceptance is False, hence this gives the first time where acceptance is False.

        Parameters
        ----------
        rejections_bool_tensor : t.Tensor
            batch_size, K
        dim : int, optional
            _description_, by default -1

        Returns
        -------
        acceptance_mask : t.Tensor
            batch_size, K
        """
        rejections_bool_tensor = ~acceptances_bool_tensor

        # This value is positive whenever there has been a rejection on the current token or before
        cumulative_rejections_tensor = t.cumsum(rejections_bool_tensor, dim=-1)
        # This value is true if there has been no rejections yet
        # In other words, we should accept every token for which this is true.
        acceptance_mask = cumulative_rejections_tensor == 0  # [batch_size, K]

        # This value is true if there has been a rejection on the current token and its the first time for this.
        first_rejection_mask = (
            cumulative_rejections_tensor == 1
        ) and rejections_bool_tensor  # [batch_size, K]

        # Everything that is not accepted or first rejected is padded
        pad_mask = t.ones_like(acceptance_mask) - acceptance_mask - first_rejection_mask

        return acceptance_mask, first_rejection_mask, pad_mask

    @staticmethod
    def _get_p_q_probabilities(
        draft_model_probs: t.Tensor,
        large_model_probs: t.Tensor,
        draft_output_ids: t.Tensor,
    ) -> Tuple[t.Tensor, t.Tensor]:
        """Get the probabilities of the chosen tokens from the draft and large models.

        Parameters
        ----------
        draft_model_probs : t.Tensor
            [batch k vocab_size]
        large_model_probs : t.Tensor
            [batch k + 1 vocab_size]
        draft_output_ids : t.Tensor
            [batch k]

        Returns
        -------
        p : t.Tensor
            [batch k]
            The probability of the drafted token from the draft model
        q: t.Tensor
            [batch k]
            The probability of the drafted token from the large model
        """
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
        )  # [batch_size, K, 1]
        p = chosen_token_draft_probs.squeeze(-1)  # [batch_size, K]

        chosen_token_large_probs = large_probs_rearranged.gather(
            -1, draft_output_ids_rearranged
        )  # [batch_size, K + 1, 1]
        q = chosen_token_large_probs.squeeze(-1)[:, :-1]  # [batch_size,K]

        return p, q

    def _check_tokens(
        self,
        draft_output_ids: t.Tensor,
        draft_model_probs: t.Tensor,
        large_model_probs: t.Tensor,
    ) -> Tuple[t.Tensor, float]:
        """Use large model to check if tokens are accepted by the rejection criteria.
        It will then keep up to before the first invalid token plus the new token from the main model.

        Parameters
        ----------
        draft_output_ids : t.Tensor
            [batch size, K]
        draft_model_probs : t.Tensor
            [batch size, K, vocab_size]
            Also called p in the paper
        large_model_probs : t.Tensor
            [batch size, K + 1, vocab_size]
            Also called q in the paper

        Returns
        -------
        accepted_output_ids: t.Tensor
            [batch size, seq len + K + 1]
        proportion_accepted: float
            Proportion of draft_tokens accepted.
        """
        batch_size, k = draft_output_ids.shape
        # k is the number of tokens we have drafted

        p, q = self._get_p_q_probabilities(
            draft_model_probs=draft_model_probs,
            large_model_probs=large_model_probs,
            draft_output_ids=draft_output_ids,
        )

        # p is the probability of the token from the draft model, q is the probability of the token from the large model
        # r is the uniform random variable which we use to decide whether to accept or reject the token
        r = t.rand_like(p)

        # Accept token if r < p / q for all tokens before current token
        acceptances = r < t.min(t.tensor(1), (p / q))  # [batch_size, K]
        acceptance_mask, first_rejection_mask, pad_mask = self.get_acceptance_mask(
            acceptances
        )
        proportion_accepted = t.sum(acceptances) / (k * batch_size)

        contrasting_model_probs = F.relu(q - p)  # [batch_size, K]
        contrast_sampled_tokens = t.multinomial(
            contrasting_model_probs, num_samples=1
        )  # [batch_size, K, 1]
        contrast_sampled_tokens = contrast_sampled_tokens.squeeze(-1)  # [batch_size, K]

        # Combine the accepted tokens if we accept with contrast_sampled_tokens if we reject
        out = (
            draft_output_ids * acceptance_mask
            + contrast_sampled_tokens * first_rejection_mask
            + pad_mask * self.pad_token_id
        )  # [batch_size, K]

        # If the final token for any batch isn't padded this means all of the tokens were accepted
        # So we sample from the large model distribution for the final token
        final_token_accepted = out[:, -1] != self.pad_token_id  # [batch_size]

        sampled_tokens = t.multinomial(
            large_model_probs[:, -1, :], num_samples=1
        )  # [batch_size, 1]
        # If the final token was accepted then we use the sampled token, otherwise we use the pad token for the k+1th token
        final_tokens = (
            sampled_tokens.squeeze(-1) * final_token_accepted
            + self.pad_token_id * ~final_token_accepted
        )  # [batch_size]

        out = t.cat([out, final_tokens.unsqueeze(-1)], dim=-1)  # [batch_size, K + 1]

        return out, proportion_accepted.item()

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
        current_max_seq_len = input_ids.shape[1]
        while current_max_seq_len < max_new_tokens:
            pre_draft_length = len(input_ids)
            # Generate K tokens
            draft_output_ids, draft_logits = self._draft_k_tokens(input_ids, K)

            # Forward pass through large model
            large_model_logits = self.large_model(
                draft_output_ids
            )  # [batch_size, seq_len + K + 1, vocab_size]

            # Softmax to get probabilities
            draft_model_probs = t.softmax(
                draft_logits / self.draft_model_temperature, dim=-1
            )  # [batch_size, seq_len + K, vocab_size]
            large_model_probs = t.softmax(
                large_model_logits / self.large_model_temperature, dim=-1
            )  # [batch_size, seq_len + K + 1, vocab_size]

            # Check if tokens are valid and keep up to before the first invalid token plus the new token from the main model
            # TODO: Design greedy decoding version as well.
            input_ids, proportion_tokens_accepted = self._check_tokens(
                draft_output_ids[:, pre_draft_length:],
                draft_model_probs[:, :pre_draft_length],
                large_model_probs[:, :pre_draft_length],
            )

            _batch, current_max_seq_len = input_ids.shape

            if verbose:
                print(current_max_seq_len, input_ids)
                print(f"Proportion of tokens accepted: {proportion_tokens_accepted}")

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
