import logging
import tempfile
import time
from functools import wraps

import fancy_einsum
import joblib
import pandas as pd
import torch as t
import transformers
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

mem = joblib.Memory(tempfile.gettempdir() + "/joblib_cache")

ACTIVATION_FUNCTIONS = dict(relu=nn.ReLU(), gelu=nn.GELU(), silu=nn.SiLU())


@mem.cache
def load_pretrained_gpt() -> GPT2LMHeadModel:
    """Load the HuggingFace GPT-2.

    On first use this downloads about 500MB from the Internet.
    Later uses should hit the cache and take under 1s to load.
    """
    return transformers.AutoModelForCausalLM.from_pretrained("gpt2")


@mem.cache
def load_pretrained_gpt_large() -> GPT2LMHeadModel:
    """Load the HuggingFace GPT-2 Large.

    On first use this downloads from the Internet.
    Later uses should hit the cache and take under 1s to load.
    """
    return transformers.AutoModelForCausalLM.from_pretrained("gpt2-large")


def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    """Assert that actual and expected are exactly equal (to floating point precision)."""
    mask = actual == expected
    if not mask.all().item():
        bad = mask.nonzero()
        msg = f"Did not match at {len(bad)} indexes: {bad[:10]}{'...' if len(bad) > 10 else ''}"
        raise AssertionError(f"{msg}\nActual:\n{actual}\nExpected:\n{expected}")


def assert_shape_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"expected shape={expected.shape}, got {actual.shape}")


def allclose(actual: t.Tensor, expected: t.Tensor, rtol=1e-4) -> bool:
    assert_shape_equal(actual, expected)
    left = (actual - expected).abs()
    right = rtol * expected.abs()
    num_wrong = (left > right).sum().item()
    if num_wrong > 0:
        print(f"Test failed. Max absolute deviation: {left.max()}")
        print(f"Actual:\n{actual}\nExpected:\n{expected}")
        raise AssertionError(
            f"allclose failed with {num_wrong} / {left.nelement()} entries outside tolerance"
        )
    print(f"Test passed with max absolute deviation of {left.max()}")
    return True


def check_leaf_nodes(model: nn.Module) -> dict[str, bool]:
    out = {}
    for p in model.named_parameters():
        out[p[0]] = p[1].is_leaf
    return out


def einsum(equation: str, *operands) -> t.Tensor:
    """Torch einsum with type hinting.

    Parameters
    ----------
    equation : str
        einsum formula which can use full words rather than just letters
    operands : Tensors to process

    Returns
    -------
    t.Tensor
    """
    return t.Tensor(fancy_einsum.einsum(equation, *operands))


def get_param_count_dict(model: nn.Module) -> pd.DataFrame:
    """Given a model return a dataframe with the layer names and counts.

    Parameters
    ----------
    model : nn.Module
        PyTorch model

    Returns
    -------
    pd.DataFrame
        layer name and parameter counts with the total (descending order)
    """
    names = []
    param_counts = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            names.append(name)
            param_counts.append(p.numel())
    names.append("Total")
    param_counts.append(sum(param_counts))
    df = pd.DataFrame({"name": names, "param_count": param_counts}).sort_values(
        by="param_count", ascending=False
    )
    return df


def tiny_stories_true_parameter_count(model: nn.Module, hidden_size: int):
    """For Tiny Stories models due to the lower vocab size and sequence length, we need to adjust the parameter count.

    Parameters
    ----------
    model : nn.Module
        PyTorch model
    hidden_size : int
        Hidden dimension size

    Returns
    -------
    int
        True parameter count
    """
    VOCAB_SIZE = 50257
    TRUE_VOCAB_SIZE = 10000
    POS_SIZE = 2048
    SEQ_LEN = 256

    total_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    unused_embedding_count = (VOCAB_SIZE - TRUE_VOCAB_SIZE) * hidden_size
    unused_positional_embedding_count = (POS_SIZE - SEQ_LEN) * hidden_size
    return total_count - unused_embedding_count - unused_positional_embedding_count


def set_logging_level(level: str = "INFO"):
    if level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        logging_level = logging.getLevelName(level)
    else:
        raise ValueError(f"Invalid logging level: {level}")
    logging.basicConfig(level=logging.INFO)
