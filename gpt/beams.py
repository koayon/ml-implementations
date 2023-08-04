from typing import Dict, List, Optional, Tuple

import tiktoken
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table

from gpt.model import FullKeyValueCache


class Beams:
    """Class to store beams during beam search.

    logprob_sums: beam
    tokens: beam seq
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: tiktoken.Encoding,
        logprob_sums: Optional[Float[t.Tensor, "beam"]],
        tokens: Optional[Int[t.Tensor, "beam seq"]],
        attention_cache: Optional[FullKeyValueCache] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eot_token = tokenizer.eot_token

        self.logprob_sums = logprob_sums
        self.tokens = tokens
        self.attention_cache = attention_cache

    def new_beams(self, logprob_sums, tokens, attention_cache) -> "Beams":
        """Creates a new Beams object with the same model and tokenizer."""
        return Beams(
            self.model,
            self.tokenizer,
            logprob_sums=logprob_sums,
            tokens=tokens,
            attention_cache=attention_cache,
        )

    def __getitem__(self, idx) -> "Beams":
        """Allows you to take a slice of the beams object along the batch dimension."""
        if (
            self.logprob_sums is None
            or self.tokens is None
            or self.attention_cache is None
        ):
            return self

        return self.new_beams(
            logprob_sums=self.logprob_sums[idx],
            tokens=self.tokens[idx],
            attention_cache=self.attention_cache[idx],
        )

    def __len__(self) -> int:
        """Returns the number of beams."""
        if self.logprob_sums is None:
            return 0
        return len(self.logprob_sums)

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

        beam_num, seq_len = self.tokens.shape

        # Forward model for next prediction
        logits, cache = self.model(
            self.tokens, cache=self.attention_cache
        )  # beam seq vocab_size

        assert cache is not None

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
        )  # beam toks seq

        tokens_for_continuations = t.cat(
            [broadcasted_prev_tokens, top_token_indices.unsqueeze(-1)], dim=-1
        )  # beam, toks_per_beam, seq + 1

        assert tokens_for_continuations.shape == (beam_num, toks_per_beam, seq_len + 1)

        # Use the new cache
        # cache list of (layer, (key, value)) tuples which contain (batch, seq, head, head_dim) tensors
        # We want to get out key of the form (batch, layer * head * head_dim * seq + 1)

        out_attention_cache = repeat(
            cache,
            "batch k_and_v layer ... -> (batch toks_per_beam) k_and_v layer ...",
            toks_per_beam=toks_per_beam,
        )

        # Flatten down to the expected dimensions with num_beam * toks_per_beam being the new batch dim
        out_logprobs = t.flatten(
            logprob_sums_for_continuations, start_dim=0, end_dim=-1
        )  # beam * toks_per_beam
        out_tokens = t.flatten(
            tokens_for_continuations, start_dim=0, end_dim=1
        )  # beam * toks_per_beam, seq + 1

        assert out_logprobs.shape == (beam_num * toks_per_beam,)
        assert out_tokens.shape == (beam_num * toks_per_beam, seq_len + 1)

        return self.new_beams(
            logprob_sums=out_logprobs,
            tokens=out_tokens,
            attention_cache=out_attention_cache,
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
        _top_log_probsums, top_indices = t.topk(
            self.logprob_sums, k=num_beams
        )  # num_beams

        # Get the associated tokens
        top_pred_tokens = self.tokens[top_indices]  # num_beams, seq

        # Check if the final tokem is an end token
        top_final_tokens = top_pred_tokens[:, -1]  # num_beams
        seq_finished_bool = top_final_tokens == self.eot_token  # num_beams
        seq_not_finished = top_final_tokens != self.eot_token  # num_beams

        # Between these two there are num_beams beams: some terminated, some not.
        best_beams = self[top_indices[seq_not_finished]]
        early_terminations = self[top_indices[seq_finished_bool]]

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
            table.add_row(f"{logprob_sum.item():>8.3f}", repr(text))
        rprint(table)
