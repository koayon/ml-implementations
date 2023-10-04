from typing import List

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

        for i, char in enumerate(character_tokens):
            self.char_to_index[char] = i + len(special_tokens)
            self.index_to_char[i + len(special_tokens)] = char

    def encode(self, text: str) -> List[int]:
        return [self.char_to_index[char] for char in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.index_to_char[token] for token in tokens])

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def batch_decode(self, tokens: List[List[int]]) -> List[str]:
        return [self.decode(token) for token in tokens]


# Example usage
tokenizer = CharTokenizer()

# Tokenizing
text = "Hello, world!"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Detokenizing
detokenized_text = tokenizer.decode(tokens)
print(f"Detokenized Text: {detokenized_text}")
