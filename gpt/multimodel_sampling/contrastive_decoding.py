import math
from typing import Tuple

import torch as t
import torch.nn as nn
from einops import einsum
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


class ContrastiveDecodingWrapper(PreTrainedModel):
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

    Reference: https://arxiv.org/pdf/2309.09117.pdf
    """

    def __init__(
        self,
        large_model: nn.Module = large_model,
        helper_models: list[nn.Module] = [helper_model],
        config=GPT2Config(),
        alpha: float = 0.1,
        temperature: float = 1.0,
        betas: list[float] = [0.5],
    ):
        super().__init__(config=config)
        self.large_model = large_model
        self.helper_models = helper_models
        self.config = config

        self.temperature = temperature

        self.alpha = alpha
        self.betas = t.tensor(betas)

        assert len(self.betas) == len(self.helper_models)

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """_summary_

        Parameters
        ----------
        x : t.Tensor
            [batch size, seq len]

        Returns
        -------
        contrastive_decoding_logits: t.Tensor
            [batch size, vocab size]
        final_main_logits: t.Tensor
            [batch size, vocab size]
        final_helper_logits: t.Tensor
            [batch size, helper model, vocab size]
        """
        batch_size, seq_len = x.shape

        # Forward pass through models
        main_logits = self.large_model(x)["logits"]  # (batch_size, seq_len, vocab_size)
        helper_logits_list = [
            helper_model(x)["logits"] for helper_model in self.helper_models
        ]  # list[num_helper_models] (batch_size, seq_len, vocab_size)

        # Combine helper model logits using beta weights
        final_helper_logit_list = [
            helper_logits[:, -1, :] for helper_logits in helper_logits_list
        ]  # list[num_helper_models] (batch_size, vocab_size)
        final_helper_logits = t.stack(
            final_helper_logit_list, dim=1
        )  # (batch_size, helper_model, vocab_size)
        combined_helper_logits = einsum(
            self.betas, final_helper_logits, "helper, batch helper vocab -> batch vocab"
        )  # (batch_size, vocab_size)

        # Define the set of vaid vocab tokens using the alpha cutoff mask
        final_main_logits = main_logits[:, -1, :]  # (batch_size, vocab_size)
        top_logit, _indices = t.max(
            final_main_logits, dim=-1
        )  # (batch_size, vocab_size)

        cutoff = math.log(self.alpha) + top_logit  # batch_size
        invalid_vocab_indices = final_main_logits < cutoff  # batch_size, vocab_size

        # Perform contrastive decoding - take difference between main and helper logits (scaled by betas)
        contrastive_decoding_logits = (
            1 + sum(self.betas)
        ) * final_main_logits - combined_helper_logits  # (batch_size, vocab_size)

        # Apply the plausibility constraint from the vocab mask
        contrastive_decoding_logits = t.masked_fill(
            contrastive_decoding_logits, invalid_vocab_indices, -t.inf
        )

        return contrastive_decoding_logits, final_main_logits, final_helper_logits

    def _generate_one_token(
        self, x: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        cd_logits, main_logits, helper_logits = self.forward(x)

        probs = t.softmax(
            cd_logits / self.temperature, dim=-1
        )  # (batch_size, vocab_size)

        # sample from probs
        sampled_output = t.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Greedy outputs for comparison
        greedy_output = t.argmax(probs, dim=-1, keepdim=True)
        main_output = t.argmax(main_logits, dim=-1, keepdim=True)
        helper_output = t.argmax(helper_logits, dim=-1, keepdim=True)

        return sampled_output, greedy_output, main_output, helper_output

    def generate(
        self, input_ids: t.Tensor, max_new_tokens: int = 5, verbose: bool = True
    ) -> t.Tensor:
        # Generate new tokens
        for i in range(max_new_tokens):
            sampled_new_token, _, _, _ = self._generate_one_token(
                input_ids
            )  # (batch_size, 1)
            if sampled_new_token == tokenizer.eos_token_id:
                break
            input_ids = t.cat(
                [input_ids, sampled_new_token], dim=-1
            )  # (batch_size, seq_len + 1)
            if verbose:
                print(i, input_ids)

        return input_ids


def main():
    cd_model = ContrastiveDecodingWrapper()
    print("Loaded models")
    input_str = "Hello, my name is"

    # Tokenize input string
    input_ids = tokenizer.encode(
        input_str, return_tensors="pt"
    )  # (batch_size, seq_len)
    input_ids = t.tensor(input_ids)  # (batch_size, seq_len)

    MODELS = {
        "CD Model": cd_model,
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
