import math
from typing import Tuple

import torch as t
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

# large_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(LARGE_MODEL)
# helper_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(HELPER_MODEL)

tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL)
large_model = load_pretrained_gpt_large()
helper_model = load_pretrained_gpt()


class ContrastiveDecodingWrapper(PreTrainedModel):
    """Contrastive Decoding approach to sampling using two models.


    Reference: https://arxiv.org/pdf/2309.09117.pdf
    """

    def __init__(
        self,
        large_model=large_model,
        helper_model=helper_model,
        config=GPT2Config(),
        alpha=0.1,
        temperature=1.0,
        beta=0.5,
    ):
        super().__init__(config=config)
        self.large_model = large_model
        self.helper_model = helper_model
        self.config = config

        self.temperature = temperature

        self.alpha = alpha
        self.beta = beta

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        batch_size, seq_len = x.shape

        main_logits = self.large_model(x)["logits"]  # (batch_size, seq_len, vocab_size)
        helper_logits = self.helper_model(x)[
            "logits"
        ]  # (batch_size, seq_len, vocab_size)

        # valid vocab indices (ones that the main/expert model will predict)

        final_helper_logit = helper_logits[:, -1, :]  # (batch_size, vocab_size)
        final_main_logit = main_logits[:, -1, :]  # (batch_size, vocab_size)
        top_logit, _indices = t.max(
            final_main_logit, dim=-1
        )  # (batch_size, vocab_size)

        # Alpha mask/cutoff
        cutoff = math.log(self.alpha) + top_logit  # batch_size
        invalid_vocab_indices = final_main_logit < cutoff  # batch_size, vocab_size

        # Perform contrastive decoding - take difference between main and helper logits (scaled by beta)
        contrastive_decoding_logits = (
            1 + self.beta
        ) * final_main_logit - self.beta * final_helper_logit

        contrastive_decoding_logits = t.masked_fill(
            contrastive_decoding_logits, invalid_vocab_indices, -t.inf
        )

        return contrastive_decoding_logits, main_logits, helper_logits

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
