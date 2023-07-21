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
DEBUG_TOLERANCES = os.getenv("DEBUG_TOLERANCES")


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
    elif DEBUG_TOLERANCES:
        print(f"Test passed with max absolute deviation of {left.max()}")


def remove_hooks(module: t.nn.Module):
    """Remove all hooks from module.

    Use module.apply(remove_hooks) to do this recursively.
    """
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()


from torch.nn.modules.module import _addindent

T = TypeVar("T")


class StaticModuleList(nn.ModuleList, Generic[T]):
    """ModuleList where the user vouches that it only contains objects of type T.

    This allows the static checker to work instead of only knowing that the contents are Modules.
    """

    # TBD lowpri: is it possible to do this just with signatures, without actually overriding the method bodies to add a cast?

    def __getitem__(self, index: int) -> T:
        return cast(T, super().__getitem__(index))

    def __iter__(self) -> Iterator[T]:
        return cast(Iterator[T], iter(self._modules.values()))

    def __repr__(self):
        # CM: modified from t.nn.Module.__repr__
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        modules = iter(self._modules.items())
        key, module = next(modules)
        n_rest = sum(1 for _ in modules)
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines + [f"+ {n_rest} more..."]

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
