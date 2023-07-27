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
        x = input_tokens.unsqueeze(0)  # batch seq

        # Initialise variables
        generated_tokens_list = []

        # Loop over length
        for _ in tqdm(range(max_tokens)):
            # Sample next token
            sampled_token = self.sample_next_token(
                input_tokens=x,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )  # batch

            out_token = sampled_token[0].item()  # int
            generated_tokens_list.append(out_token)

            if stop_at_end_token and out_token == self.eot_token:
                break

        return self.tokenizer.decode(generated_tokens_list)

    def generate_beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
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
        )

        collected_beams = Beams(
            model=self.model,
            tokenizer=self.tokenizer,
            logprob_sums=None,
            tokens=None,
        )

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
            collected_beams += best_terminated_beams
            num_beams -= len(best_terminated_beams)

            # Continue with the best continuing beams
            beams = best_continuing_beams

            # If all the beams terminated then stop
            if num_beams == 0:
                break

        # Add the best beams to the collected beams
        collected_beams += beams

        # Choose the best token completion by logprob sum
        logprobs_and_completions = collected_beams.logprobs_and_completions
        best_logprob_sum = max(logprobs_and_completions.keys())
        best_completion = logprobs_and_completions[best_logprob_sum]

        return best_completion


class Beams:
    """Class to store beams during beam search."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: tiktoken.Encoding,
        logprob_sums: Optional[Float[t.Tensor, "beam"]],
        tokens: Optional[Int[t.Tensor, "beam seq"]],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eot_token = tokenizer.eot_token

        self.logprob_sums = logprob_sums
        self.tokens = tokens

    def new_beams(self, logprob_sums, tokens) -> "Beams":
        """Creates a new Beams object with the same model and tokenizer."""
        return Beams(
            self.model,
            self.tokenizer,
            logprob_sums,
            tokens,
        )

    def __getitem__(self, idx) -> "Beams":
        """Allows you to take a slice of the beams object along the batch dimension."""
        if self.logprob_sums is None or self.tokens is None:
            return self
        return self.new_beams(self.logprob_sums[idx], self.tokens[idx])

    def __len__(self) -> int:
        """Returns the number of beams."""
        if self.logprob_sums is None:
            return 0
        return len(self.logprob_sums)

    def __add__(self, other) -> "Beams":
        """Combines two beams objects."""
        if self.logprob_sums is None or self.tokens is None:
            return other

        return self.new_beams(
            t.cat([self.logprob_sums, other.logprob_sums], dim=0),  # beam
            t.cat(
                [self.tokens, other.tokens], dim=0
            ),  # beam seq (seqs may be different lengths)
        )

    @property
    def logprobs_and_completions(self) -> Dict[float, str]:
        """Returns self as a list of logprob sums and completions (useful for getting final output)."""
        if self.tokens is None or self.logprob_sums is None:
            return {}

        return {
            logprob_sum.item(): self.tokenizer.decode(tokens.numpy().tolist())
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        }

    def generate(
        self, toks_per_beam: int, no_repeat_ngram_size: Optional[int] = None
    ) -> "Beams":
        """
        Starting from the current set of beams (which has length `num_beams`), returns a new
        set of `num_beams * toks_per_beam`, containing the best `toks_per_beam` continuations for each
        of the original beams.
        """

        if self.tokens is None:
            return self

        # Forward model for next prediction
        logits = self.model(self.tokens)  # beam seq vocab_size
        next_token_logits = logits[:, -1, :]  # beam vocab_size

        # Get log_softmax of next token logits. We use this so they are comparable across beams.
        next_token_logprobs = F.log_softmax(
            next_token_logits, dim=-1
        )  # beam vocab_size

        # Restrict to top_k next predictions
        top_token_logprobs, top_token_indices = t.topk(
            next_token_logprobs, k=toks_per_beam, dim=-1
        )  # beam toks_per_beam

        # Add logprobs of continuations to logprobs of beams. (First repeat logprob_sums to match shapes for addition.)
        broadcasted_logprob_sums = repeat(
            self.logprob_sums, "beam -> beam toks_per_beam", toks_per_beam=toks_per_beam
        )  # beam toks_per_beam
        logprob_sums_for_continuations = (
            broadcasted_logprob_sums + top_token_logprobs
        )  # beam, toks_per_beam

        # Concatenate our new token predictions onto the end of the completions
        broadcasted_prev_tokens = repeat(
            self.tokens,
            "beam seq -> beam toks_per_beam seq",
            toks_per_beam=toks_per_beam,
        )

        tokens_for_continuations = t.stack(
            [broadcasted_prev_tokens, top_token_indices], dim=-1
        )  # beam, toks_per_beam, seq + 1

        # Flatten down to the expected dimensions with num_beam * toks_per_beam being the new batch dim
        out_logprobs = t.flatten(
            logprob_sums_for_continuations, start_dim=0, end_dim=-1
        )  # beam * toks_per_beam
        out_tokens = t.flatten(
            tokens_for_continuations, start_dim=0, end_dim=1
        )  # beam * toks_per_beam, seq + 1

        return self.new_beams(
            logprob_sums=logprob_sums_for_continuations, tokens=top_token_indices
        )

        # TODO: Implement no_repeat_ngram_size

    def filter(
        self,
        num_beams: int,
    ) -> Tuple["Beams", "Beams"]:
        """
        self.logprob_sums: beam * toks_per_beam
        self.tokens: beam * toks_per_beam, seq + 1

        Returns:
            best_beams: Beams
                filtered version of self, containing top `num_beams` which are also not terminated.

            early_terminations: Beams
                filtered version of self, containing top `num_beams` which are also terminated.
                i.e. the sum of lengths of these two should equal `num_beams`.
        """
        if self.tokens is None or self.logprob_sums is None:
            return self, self

        # Get top "num_beams" of the larger set of beams
        top_log_probsums, top_indices = t.topk(
            self.logprob_sums, k=num_beams
        )  # num_beams

        # Get the associated tokens
        top_pred_tokens = self.tokens[top_indices]  # num_beams, seq

        # Check if the final tokem is an end token
        top_final_tokens = top_pred_tokens[:, -1]  # num_beams
        seq_finished_bool = top_final_tokens == self.eot_token  # num_beams
        seq_not_finished = top_final_tokens != self.eot_token  # num_beams

        # Between these two there are num_beams beams: some terminated, some not
        best_beams = self[seq_not_finished]
        early_terminations = self[seq_finished_bool]

        return best_beams, early_terminations

    def print(self, title="Best completions", max_print_chars=80) -> None:
        """
        Prints out a set of sequences with their corresponding logitsums.
        """
        if self.tokens is None or self.logprob_sums is None:
            return

        if len(self.tokens) == 0:
            return

        table = Table("logitsum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens.numpy().tolist())
            if len(repr(text)) > max_print_chars:
                text = (
                    text[: int(0.3 * max_print_chars)]
                    + " ... "
                    + text[-int(0.7 * max_print_chars) :]
                )
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)


if __name__ == "__main__":
    pass
