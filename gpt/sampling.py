import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tiktoken
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from tqdm.auto import tqdm

from gpt.beams import Beams
from gpt.cached_attention import AttentionCache
from gpt.config import GPTConfig
from gpt.model import GPT2


class TransformerSampler:
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model("gpt2"),
    ):
        self.model = model or GPT2(with_pretrained_weights=True)
        self.config = self.model.config

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
        cache: Optional[List[AttentionCache]] = None,
        *,
        top_k: int,
        top_p: float,
        temperature: float = 1.0,
    ) -> Tuple[t.Tensor, List[AttentionCache]]:
        """Given a sequence of tokens, generate the next one.

        input_tokens: batch, seq
        Returns: batch
        """

        # Forward pass tokens
        with t.inference_mode():
            with t.no_grad():
                all_logits, cache = self.model(
                    input_tokens, cache
                )  # batch seq vocab_size

        if cache is None:
            cache = []

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
                return sampled_tokens, cache

        if top_p:
            top_logits, list_set_top_p_indices = self._logits_to_top_p(
                logits, top_p=top_p
            )  # batch k

            if not top_k:
                sampled_tokens = self._sample_with_temp(
                    top_logits, temperature=temperature
                )  # batch
                return sampled_tokens, cache

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

            return sampled_tokens, cache

        # If no top_p or top_k, sample from logits (basic categorical sampling)
        sampled_tokens = self._sample_with_temp(
            logits, temperature=temperature
        )  # batch

        return sampled_tokens, cache

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        stop_at_end_token: bool = True,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> str:
        """Generate a sequence of tokens.

        Returns:
        generated_output: str
        """

        # Tokenise input
        input_tokens = t.tensor(self.tokenizer.encode(prompt))  # seq
        x = input_tokens.unsqueeze(0)  # 1 seq

        # Initialise variables
        generated_tokens_list = []
        cache = None

        # Loop over length
        for _timestep in tqdm(range(max_tokens)):
            st_time = time.time()
            # Sample next token
            sampled_token, cache = self.sample_next_token(
                input_tokens=x,
                cache=cache,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )  # batch

            end_time = time.time()
            print(f"Time taken for sample {_timestep}: {end_time - st_time}")

            out_token = sampled_token[0].item()  # int
            generated_tokens_list.append(out_token)
            x = t.cat([x, sampled_token.unsqueeze(0)], dim=1)  # batch seq

            if stop_at_end_token and out_token == self.eot_token:
                break

        return self.tokenizer.decode(generated_tokens_list)

    def generate_beam_search(
        self,
        prompt: str,
        max_new_tokens: int,
        num_return_sequences: Optional[int] = None,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 0,
    ) -> str:
        """Generate a sequence of tokens using beam search.

        Returns:
        generated_output: str
        """
        # Tokenise input
        input_tokens = t.tensor(self.tokenizer.encode(prompt))  # seq
        x = input_tokens.unsqueeze(0)  # 1 seq

        # assert num_return_sequences <= num_beams
        self.model.eval()

        beams = Beams(
            model=self.model,
            tokenizer=self.tokenizer,
            logprob_sums=t.zeros(x.shape[0]),  # batch
            tokens=x,
            attention_cache=None,
        )

        collected_beams = [
            Beams(
                model=self.model,
                tokenizer=self.tokenizer,
                logprob_sums=None,
                tokens=None,
                attention_cache=None,
            )
        ]

        for i in range(max_new_tokens):
            # Generate new beams branching off from the current set of beams
            generated_beams = beams.generate(
                toks_per_beam=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
            )

            # Print results
            generated_beams.print(title=f"Generated beams - step {i}")

            # Filter the new beams to get the best `num_beams` continuations
            best_continuing_beams, best_terminated_beams = generated_beams.filter(
                num_beams=num_beams
            )

            # Add the best terminated continuations to the current set of beams
            collected_beams += [best_terminated_beams]
            num_beams -= len(best_terminated_beams)

            # Continue with the best continuing beams
            beams = best_continuing_beams

            # If all the beams terminated then stop
            if num_beams == 0:
                break

        # Add the best beams to the collected beams
        collected_beams += [beams]

        # Choose the best token completion by logprob sum
        collected_beams_dict = {}
        for beam in collected_beams:
            collected_beams_dict.update(beam.logprobs_and_completions)

        best_logprob_sum = max(collected_beams_dict.keys())
        best_completion = collected_beams_dict[best_logprob_sum]

        return best_completion


if __name__ == "__main__":
    model_sampler = TransformerSampler()

    PROMPT = "Paul Graham is an English computer scientist, essayist, entrepreneur, venture capitalist, and author. He is best known for his work on the programming language Lisp, his former startup Viaweb (later renamed Yahoo! Store), cofounding the influential startup accelerator and seed capital firm Y Combinator, his essays, and Hacker News. He is the author of several computer programming books, including:"

    # sampled_tokens = model_sampler.generate_beam_search(
    #     prompt=PROMPT,
    #     max_new_tokens=5,
    # )
    sampled_tokens = model_sampler.generate(
        prompt=PROMPT,
        max_tokens=100,
    )
    print(sampled_tokens)
