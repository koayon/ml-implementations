import pytest
import torch as t

from gpt.multimodel_sampling.speculative_decoding import SpeculativeDecodingWrapper

sd_model = SpeculativeDecodingWrapper()

VOCAB_SIZE = 50257


class TestGetAcceptanceMask:
    # Acceptances bool tensor is all True
    def test_all_true(self):
        acceptances_bool_tensor = t.tensor([[True, True, True], [True, True, True]])
        (
            acceptance_mask,
            first_rejection_mask,
            pad_mask,
        ) = SpeculativeDecodingWrapper.get_acceptance_mask(acceptances_bool_tensor)
        assert acceptance_mask.all()
        assert not first_rejection_mask.any()
        assert not pad_mask.any()

    # Acceptances bool tensor is all False
    def test_all_false(self):
        acceptances_bool_tensor = t.tensor(
            [[False, False, False], [False, False, False]]
        )
        (
            acceptance_mask,
            first_rejection_mask,
            pad_mask,
        ) = SpeculativeDecodingWrapper.get_acceptance_mask(acceptances_bool_tensor)
        assert not acceptance_mask.any()
        assert (
            first_rejection_mask
            == t.tensor([[True, False, False], [True, False, False]])
        ).all()
        assert (pad_mask == t.tensor([[False, True, True], [False, True, True]])).all()

    # Acceptances bool tensor is random
    def test_random(self):
        acceptances_bool_tensor = t.tensor([[True, False, True], [False, True, False]])
        (
            acceptance_mask,
            first_rejection_mask,
            pad_mask,
        ) = SpeculativeDecodingWrapper.get_acceptance_mask(acceptances_bool_tensor)

        assert (
            acceptance_mask == t.tensor([[True, False, False], [False, False, False]])
        ).all()
        assert (
            first_rejection_mask
            == t.tensor([[False, True, False], [True, False, False]])
        ).all()
        assert (pad_mask == t.tensor([[False, False, True], [False, True, True]])).all()

        # TODO: Separate out expected values from test


def test_forward():
    pass
