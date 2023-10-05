from typing import List, Optional

CHARACTER_VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.'!? +-*/=()[]<>%^&*~"


class CharTokenizer:
    def __init__(
        self,
        special_tokens: List[str] = ["<PAD>", "<SOS>", "<EOS>", "<IDK>"],
        character_tokens: str = CHARACTER_VOCAB,
    ):
        self.special_tokens = special_tokens
        self.char_to_index: dict[str, int] = {}
        self.index_to_char: dict[int, str] = {}

        for i, token in enumerate(special_tokens):
            self.char_to_index[token] = i
            self.index_to_char[i] = token

        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.idk_token = "<IDK>"

        self.pad_token_id = self.char_to_index[self.pad_token]
        self.sos_token_id = self.char_to_index[self.sos_token]
        self.eos_token_id = self.char_to_index[self.eos_token]
        self.idk_token_id = self.char_to_index[self.idk_token]

        for i, char in enumerate(character_tokens):
            self.char_to_index[char] = i + len(special_tokens)
            self.index_to_char[i + len(special_tokens)] = char

    def __len__(self):
        return len(self.char_to_index)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_index[char] for char in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.index_to_char[token] for token in tokens])

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def batch_decode(self, tokens: List[List[int]]) -> List[str]:
        return [self.decode(token) for token in tokens]

    def pad(
        self,
        encoding: List[List[int]],
        max_length: Optional[int] = None,
        padding_value: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors="pt",
        padding: bool = True,
    ) -> List[List[int]]:
        # if max_length is None:
        #     max_length = max(len(seq) for seq in encoding)
        # if padding_value is None:
        #     padding_value = self.pad_token_id

        # return [seq + [padding_value] * (max_length - len(seq)) for seq in encoding]

        return encoding


def main():
    # Example usage
    tokenizer = CharTokenizer()

    # Tokenizing
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    print(f"Tokens: {tokens}")

    # Detokenizing
    detokenized_text = tokenizer.decode(tokens)
    print(f"Detokenized Text: {detokenized_text}")


if __name__ == "__main__":
    main()
