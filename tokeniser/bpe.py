from collections import Counter
from dataclasses import dataclass, field


def str_to_utf8_tokens(text: str, verbose: bool = True) -> list[int]:
    byte_string = text.encode("utf-8")
    tokens = list(map(int, byte_string))

    if verbose:
        print(f"Text: {len(text)} characters - {text[:50]}...")
        print(f"Tokens: {len(tokens)} bytes - {tokens[:50]}...")

    return tokens


INITIAL_VOCAB: dict[str, int] = {chr(i): i for i in range(256)}


@dataclass
class BPE:
    initial_vocab: dict[str, int] = field(default_factory=lambda: INITIAL_VOCAB)
    merges: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self):
        self.vocab = self.initial_vocab
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError

    def train(self, text: str, max_vocab_size: int) -> None:
        tokens = str_to_utf8_tokens(text)

        original_vocab_size = self.vocab_size
        original_tokens_length = len(tokens)

        while self.vocab_size < max_vocab_size:
            tokens = self.merge_step(tokens)

        final_vocab_size = self.vocab_size
        final_tokens_length = len(tokens)

        print("Training complete!")
        print(f"Vocab size: {final_vocab_size} from {original_vocab_size}")
        print(f"Tokens length: {final_tokens_length} from {original_tokens_length}")

        print(
            f"Overall this is a compression ratio of {final_tokens_length/original_tokens_length:.2f}x"
        )

    def merge_step(self, tokens: list[int]) -> list[int]:
        most_common_pair, _top_count = self.get_byte_pairs(tokens).most_common(1)[0]
        new_token_id = self.mint_token(most_common_pair)

        print(f"Merging: {most_common_pair} -> {new_token_id}")

        updated_tokens: list[int] = []

        idx = 0
        while idx < len(tokens) - 1:
            if (tokens[idx], tokens[idx + 1]) == most_common_pair:
                updated_tokens.append(new_token_id)
                idx += 2
            else:
                updated_tokens.append(tokens[idx])
                idx += 1

        if updated_tokens[-1] != new_token_id:
            updated_tokens.append(tokens[-1])

        return updated_tokens

    def mint_token(self, tokens_to_merge: tuple[int, int]) -> int:
        raise NotImplementedError

    def get_byte_pairs(self, tokens: list[int]) -> Counter[tuple[int, int]]:
        return Counter(zip(tokens[:-1], tokens[1:]))
