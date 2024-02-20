from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


def str_to_utf8_tokens(text: str, verbose: bool = True) -> list[int]:
    byte_string = text.encode("utf-8")
    tokens = list(map(int, byte_string))

    if verbose:
        print(f"Text: {len(text)} characters - {text[:50]}...")
        print(f"Tokens: {len(tokens)} bytes - {tokens[:50]}...")

    return tokens


INITIAL_VOCAB: dict[str, int] = {chr(i): i for i in range(256)}
