import math
from typing import Tuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from coverage import FileReporter
from einops import einsum, rearrange, repeat
from jaxtyping import Int
from spacy import vocab
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
tokenizer.pad_token = "<PAD>"
tokenizer.pad_token_id = len(tokenizer) - 2

large_model = load_pretrained_gpt_large()
helper_model = load_pretrained_gpt()

PAD_TOKEN_ID = tokenizer.pad_token_id


def sample(
    sampling_weights: t.Tensor, num_samples: int = 1
) -> Int[t.Tensor, "batch_size seq_len num_samples"]:
    if sampling_weights.ndim == 2:
        sampling_weights = sampling_weights.unsqueeze(1)

    batch_size, seq_len, vocab_size = sampling_weights.shape

    flat_sampling_weights = rearrange(
        sampling_weights, "batch seq_len vocab -> (batch seq_len) vocab"
    )
    sample_ids = t.multinomial(
        flat_sampling_weights, num_samples=num_samples
    )  # [batch_size * seq_len, num_samples]

    sample_ids = rearrange(
        sample_ids,
        "(batch seq_len) num_samples -> batch seq_len num_samples",
        batch=batch_size,
        seq_len=seq_len,
    )

    return sample_ids


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

        eps = 1e-6

        # Add epsilon to avoid dividing by zero
        self.large_model_temperature = large_model_temperature + eps
        self.draft_model_temperature = draft_model_temperature + eps
        self.K = K

        assert K > 0
        assert 0 <= large_model_temperature <= 2
        assert 0 <= draft_model_temperature <= 2

    def forward(self, input_ids: t.Tensor) -> t.Tensor:
        """Forward with the main model

        Parameters
        ----------
        x : t.Tensor
            [batch size, seq len]

        """
        return self.large_model(input_ids)

    def get_attention_mask(
        self, last_non_pad_token_per_batch: t.Tensor, seq_len: int
    ) -> t.Tensor:
        """Get the attention mask for the current tokens.

        Parameters
        ----------
        last_non_pad_token_per_batch : t.Tensor
            [batch_size]

        Returns
        -------
        attention_mask : t.Tensor
            [batch_size, seq_len]
        """
        assert last_non_pad_token_per_batch.ndim == 1

        if t.min(last_non_pad_token_per_batch) < 0:
            raise ValueError(
                "last_non_pad_token_per_batch should be a tensor of non-negative integers"
            )

        batch_size = last_non_pad_token_per_batch.shape[0]

        range = t.arange(seq_len)
        expanded_range = repeat(
            range, "seq_len -> batch seq_len", batch=batch_size
        )  # [batch_size, seq_len]

        expanded_last_tok = repeat(
            last_non_pad_token_per_batch, "batch -> batch seq_len", seq_len=seq_len
        )

        attention_mask = expanded_range <= expanded_last_tok

        return attention_mask

    def _draft_k_tokens(
        self, input_ids: t.Tensor, last_non_pad_token_per_batch: t.Tensor, K: int
    ) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
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
        last_non_pad_token_per_batch:
            batch_size
        """

        helper_logits = t.empty((1))
        batch_size, seq_len = input_ids.shape

        for i in range(K):
            batch_size, seq_len = input_ids.shape

            # print(last_non_pad_token_per_batch)
            attention_mask = self.get_attention_mask(
                last_non_pad_token_per_batch, seq_len
            )

            # print(attention_mask.shape)
            # print("attention_mask", attention_mask)

            # Forward pass through helper model
            helper_out = self.helper_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            helper_logits = helper_out.logits  # [batch_size, seq_len + i, vocab_size]

            # Greedy sample
            # TODO: Switch to multinomial sampling
            sampled_new_token = t.argmax(
                helper_logits[:, -1, :],
                dim=-1,
            ).unsqueeze(
                0
            )  # [batch_size, 1]

            # If the new token is a pad token then we don't want to increment the last_non_pad_token_per_batch
            last_non_pad_token_per_batch += (
                sampled_new_token.squeeze() != PAD_TOKEN_ID
            )  # [batch_size]

            input_ids = t.cat(
                [input_ids, sampled_new_token], dim=-1
            )  # [batch_size, seq_len + 1]

            # Add to the attention_mask to account for the new token

            if (input_ids[:, -1] == tokenizer.eos_token_id).all():
                # Early stopping if all sequences have ended
                break

        draft_output_ids = input_ids  # [batch_size, seq_len + K]

        # Pad the logits with zeros to make the shape compatible with the ids
        zeros = t.zeros_like(helper_logits[:, 0:1, :])  # [batch_size, 1, vocab_size]
        helper_logits = t.cat(
            [zeros, helper_logits], dim=1
        )  # [batch_size, seq_len + K, vocab_size]

        assert seq_len < draft_output_ids.shape[1] <= seq_len + K

        return draft_output_ids, helper_logits, last_non_pad_token_per_batch

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
        ) * rejections_bool_tensor  # [batch_size, K]

        pad_mask = t.ones_like(acceptance_mask, dtype=t.bool)

        # Everything that is not accepted or first rejected is padded
        pad_mask = t.logical_xor(pad_mask, acceptance_mask)
        pad_mask = t.logical_xor(pad_mask, first_rejection_mask)

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
            [batch k vocab_size]
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
            draft_model_probs, "batch k vocab -> batch k 1 vocab"
        )
        large_probs_rearranged = rearrange(
            large_model_probs, "batch k vocab -> batch k 1 vocab"
        )
        draft_output_ids_rearranged = rearrange(
            draft_output_ids, "batch k -> batch k 1 1"
        )

        # Gather the probabilities of the chosen tokens
        chosen_token_draft_probs = draft_probs_rearranged.gather(
            -1, draft_output_ids_rearranged
        )  # [batch_size, K, 1]
        p = chosen_token_draft_probs.squeeze(-1).squeeze(-1)  # [batch_size, K]

        chosen_token_large_probs = large_probs_rearranged.gather(
            -1, draft_output_ids_rearranged
        )  # [batch_size, K + 1, 1]
        q = chosen_token_large_probs.squeeze(-1).squeeze(-1)  # [batch_size,K]

        assert (
            p.shape == q.shape
        ), f"p and q should have the same shape, but got {p.shape} and {q.shape}"

        return p, q

    def _check_tokens(
        self,
        draft_output_ids: t.Tensor,
        draft_model_probs: t.Tensor,
        large_model_probs: t.Tensor,
    ) -> Tuple[t.Tensor, t.Tensor, t.Tensor, float]:
        """Use large model to check if tokens are accepted by the rejection criteria.
        It will then keep up to before the first invalid token.
        For the first invalid token if one exists, we use the contrastive/rejection sampling approach.

        Parameters
        ----------
        draft_output_ids : t.Tensor
            [batch size, K]
        draft_model_probs : t.Tensor
            [batch size, K, vocab_size]
            Also called p in the paper
        large_model_probs : t.Tensor
            [batch size, K, vocab_size]
            Also called q in the paper

        Returns
        -------
        accepted_output_ids: t.Tensor
            [batch size, seq len + K]
        final_token_accepted: t.Tensor
            bool [batch size]
            Whether or not the final token was accepted for each batch.
            If yes we need to sample from the large model distribution for the final token.
        num_tokens_accepted: t.Tensor
            [batch size]
            The number of tokens accepted for each batch.
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
        p_q_ratio = t.min(t.tensor(1), (p / q))  # [batch_size, K]

        acceptances = r < p_q_ratio  # [batch_size, K]

        acceptance_mask, first_rejection_mask, pad_mask = self.get_acceptance_mask(
            acceptances
        )
        num_tokens_accepted = t.sum(acceptance_mask, dim=-1)  # [batch_size]
        proportion_accepted = t.sum(acceptances) / (k * batch_size)

        contrasting_model_probs = F.relu(
            large_model_probs - draft_model_probs
        )  # [batch_size, K, vocab_size]

        contrast_sampled_tokens = sample(contrasting_model_probs)  # [batch_size, K, 1]
        contrast_sampled_tokens = contrast_sampled_tokens.squeeze(-1)  # [batch_size, K]

        # Combine the accepted tokens if we accept with contrast_sampled_tokens if we reject

        print("acceptances", acceptances)
        print("draft_output_ids", draft_output_ids * acceptance_mask.squeeze(-1))
        print(
            "contrast_sampled_tokens",
            contrast_sampled_tokens * first_rejection_mask.squeeze(-1),
        )
        print("PAD_TOKEN_ID_ids", PAD_TOKEN_ID * pad_mask.squeeze(-1))

        out = (
            draft_output_ids * acceptance_mask.squeeze(-1)
            + contrast_sampled_tokens * first_rejection_mask.squeeze(-1)
            + PAD_TOKEN_ID * pad_mask.squeeze(-1)
        )  # [batch_size, K]

        print("out", out)

        assert out.shape == (batch_size, k)

        # If the final token for any batch isn't padded this means all of the tokens were accepted
        # So we sample from the large model distribution for the final token
        final_token_accepted = ~pad_mask[:, -1]  # bool [batch_size]

        return (
            out,
            final_token_accepted,
            num_tokens_accepted,
            proportion_accepted.item(),
        )

    def sample_final_token(
        self, large_model_probs: t.Tensor, final_token_accepted: t.Tensor
    ) -> t.Tensor:
        """If the final token was accepted then we use the sampled token, otherwise we use the pad token for the k+1th token.

        Parameters
        ----------
        large_model_probs : t.Tensor
            [batch_size, seq_len + K + 1, vocab_size]
        final_token_accepted : t.Tensor
            bool [batch_size]

        Returns
        -------
        final_tokens : t.Tensor
            [batch_size, 1]
        """

        batch_size, seq_len, vocab_size = large_model_probs.shape

        final_sampled_tokens = sample(large_model_probs[:, -1, :])  # [batch_size, 1, 1]

        final_sampled_tokens = rearrange(
            final_sampled_tokens, "batch 1 1 -> batch"
        )  # [batch_size]

        assert (
            final_sampled_tokens.shape == final_token_accepted.shape
        ), f"final_sampled_tokens and final_token_accepted should have the same shape, but got {final_sampled_tokens.shape} and {final_token_accepted.shape}"

        # If the final token was accepted then we use the sampled token, otherwise we use the pad token for the k+1th token
        final_tokens = (
            final_sampled_tokens * final_token_accepted
            + PAD_TOKEN_ID * ~final_token_accepted
        )  # [batch_size]
        final_tokens = final_tokens.unsqueeze(-1)  # [batch_size, 1]

        assert final_tokens.shape == (
            batch_size,
            1,
        ), f"final_tokens should have shape (batch_size, 1), but got {final_tokens.shape}"

        return final_tokens  # [batch_size, 1]

    @staticmethod
    def get_k_last_tokens(
        input_tensor: t.Tensor, last_non_pad_token_per_batch: t.Tensor, K: int
    ) -> t.Tensor:
        """Get the last K tokens for each batch in the input tensor.

        Parameters
        ----------
        input_tensor : t.Tensor
            [batch_size, seq_len] or [batch_size, seq_len, vocab_size]
        last_non_pad_token_per_batch : t.Tensor
            [batch_size]
        K : int
            The number of tokens back to get.

        Returns
        -------
        t.Tensor
            [batch_size, K] or [batch_size, K, vocab_size

        Raises
        ------
        ValueError
            If the input_tensor has more than 3 dimensions.
        """
        assert last_non_pad_token_per_batch.ndim == 1
        assert K > 0

        if input_tensor.ndim == 2:
            filtering_tensor = repeat(
                last_non_pad_token_per_batch, "batch -> batch k", k=K
            )  # [batch_size, K]
            filtering_tensor = filtering_tensor - K + t.arange(K) + 1  # [batch_size, K]

        elif input_tensor.ndim == 3:
            vocab_size = input_tensor.shape[-1]

            filtering_tensor = repeat(
                last_non_pad_token_per_batch,
                "batch -> batch k vocab",
                k=K,
                vocab=vocab_size,
            )  # [batch_size, K]

            filtering_tensor = (
                filtering_tensor
                - K
                + repeat(t.arange(K), "k -> k vocab", vocab=vocab_size)
                + 1
            )  # [batch_size, K vocab_size]
        else:
            raise ValueError(
                f"input_tensor should have 2 or 3 dimensions, but got {input_tensor.ndim} dims"
            )

        out = input_tensor.gather(
            1, filtering_tensor
        )  # [batch_size, K] or [batch_size, K, vocab_size]

        return out

    def generate(
        self,
        input_ids: t.Tensor,
        max_new_tokens: int = 20,
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
        initial_seq_len = input_ids.shape[1]
        tokens_added = 0
        output_ids = t.empty((1))

        # If a row doesn't contain the pad token then set the value to be len(input_ids) - 1
        # If it does then set the value to be that token index
        high_pad = (input_ids == PAD_TOKEN_ID).float() * 1e9 + (
            input_ids != PAD_TOKEN_ID
        ).float() * t.arange(input_ids.shape[1])
        last_non_pad_token_per_batch = t.argmax(high_pad, dim=-1)  # [batch_size]

        while tokens_added < max_new_tokens:
            if verbose:
                print("Hit start of while loop!")
                print("-------------------------")
                print(f"Completion so far at {tokens_added} tokens added:")
                print(
                    tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True)
                )

            # Step 1: Draft Generate K tokens
            (
                draft_output_ids,
                draft_logits,
                last_non_pad_token_per_batch,
            ) = self._draft_k_tokens(
                input_ids=input_ids,
                last_non_pad_token_per_batch=last_non_pad_token_per_batch,
                K=K,
            )

            if verbose:
                print("Drafted sentence: ")
                print(
                    tokenizer.batch_decode(
                        draft_output_ids.tolist(), skip_special_tokens=True
                    )
                )

            # Step 2: Forward pass through large model
            attention_mask = self.get_attention_mask(
                last_non_pad_token_per_batch, seq_len=draft_output_ids.shape[1]
            )

            large_model_output = self.large_model(
                input_ids=draft_output_ids, attention_mask=attention_mask
            )
            large_model_logits = (
                large_model_output.logits
            )  # [batch_size, seq_len + K, vocab_size]

            # Pad the logits with zeros to make the shape compatible with the ids
            zeros = t.zeros_like(
                large_model_logits[:, 0:1, :]
            )  # [batch_size, 1, vocab_size]
            large_model_logits = t.cat(
                [zeros, large_model_logits], dim=1
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
            # TODO: Application to beam search

            # Step 3: Based on acceptance criteria, check whether to accept tokens or not
            (
                output_ids,
                final_token_accepted,
                num_tokens_accepted,
                proportion_tokens_accepted,
            ) = self._check_tokens(
                self.get_k_last_tokens(
                    draft_output_ids,
                    last_non_pad_token_per_batch=last_non_pad_token_per_batch,
                    K=K,
                ),
                self.get_k_last_tokens(
                    draft_model_probs,
                    last_non_pad_token_per_batch=last_non_pad_token_per_batch,
                    K=K,
                ),
                self.get_k_last_tokens(
                    large_model_probs[:, :-1],
                    last_non_pad_token_per_batch=last_non_pad_token_per_batch,
                    K=K,
                ),
            )

            # TODO: We might need to make all the batches the same size here so we're looking at, the min number of tokens accepted (perhaps with some lenience)

            # Step 4: Sample the final token from the large model distribution if it was accepted
            final_tokens = self.sample_final_token(
                large_model_probs=large_model_probs,
                final_token_accepted=final_token_accepted,
            )
            last_non_pad_token_per_batch += final_token_accepted

            output_ids = t.cat(
                [output_ids, final_tokens], dim=-1
            )  # [batch_size, seq_len + K + 1]

            if (output_ids[:, -1] == tokenizer.eos_token_id).all():
                # Early stopping if all sequences have ended
                print("Break!")
                break

            # Step 5: Add the new tokens to the input_ids

            input_ids = t.cat(
                [input_ids, output_ids], dim=-1
            )  # [batch_size, seq_len + K + 1]

            tokens_added += K + 1

            if verbose:
                print("seq_shape", input_ids.shape)
                print(f"Proportion of tokens accepted: {proportion_tokens_accepted}")
                print("last_non_pad_token_per_batch", last_non_pad_token_per_batch)

        output_ids = input_ids
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
