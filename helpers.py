import http
import logging
import os
import tempfile
import time
from functools import wraps
from typing import Callable, Generic, Iterator, Optional, TypeVar, cast

import joblib
import requests
import torch as t
import transformers
from torch import nn
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

mem = joblib.Memory(tempfile.gettempdir() + "/joblib_cache")

ACTIVATION_FUNCTIONS = dict(relu=nn.ReLU(), gelu=nn.GELU())


@mem.cache
def load_pretrained_gpt() -> GPT2LMHeadModel:
    """Load the HuggingFace GPT-2.

    On first use this downloads about 500MB from the Internet.
    Later uses should hit the cache and take under 1s to load.
    """
    return transformers.AutoModelForCausalLM.from_pretrained("gpt2")


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


def allclose(actual: t.Tensor, expected: t.Tensor, rtol=1e-4) -> None:
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


def remove_hooks(module: t.nn.Module):
    """Remove all hooks from module.

    Use module.apply(remove_hooks) to do this recursively.
    """
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()


def check_leaf_nodes(model: nn.Module) -> dict[str, bool]:
    out = {}
    for p in model.named_parameters():
        out[p[0]] = p[1].is_leaf
    return out
