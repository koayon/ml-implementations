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


class TestGetAttentionMask:
    # Returns a tensor of shape [batch_size, seq_len] when given a tensor of shape [batch_size].
    def test_returns_attention_mask_with_correct_shape(self):
        last_non_pad_token_per_batch = t.tensor([3, 5, 2])
        model = SpeculativeDecodingWrapper()
        attention_mask = model.get_attention_mask(last_non_pad_token_per_batch)
        assert attention_mask.shape == (3, 6)

        assert (
            attention_mask
            == t.tensor(
                [
                    [True, True, True, True, False, False],
                    [True, True, True, True, True, True],
                    [True, True, True, False, False, False],
                ]
            )
        ).all()

    # Returns a tensor of all True values when given a tensor of shape [batch_size] with all values equal to seq_len - 1.
    def test_returns_attention_mask_with_all_true_values(self):
        last_non_pad_token_per_batch = t.tensor([5, 5, 5])
        model = SpeculativeDecodingWrapper()
        attention_mask = model.get_attention_mask(last_non_pad_token_per_batch)
        assert attention_mask.all()

    # Returns a tensor of one True and the rest False when given a tensor of shape [batch_size] with all values equal to 0.
    def test_returns_attention_mask_with_all_false_values(self):
        last_non_pad_token_per_batch = t.tensor([0, 2, 0])
        model = SpeculativeDecodingWrapper()
        attention_mask = model.get_attention_mask(last_non_pad_token_per_batch)
        assert (
            attention_mask
            == t.tensor(
                [
                    [True, False, False],
                    [True, True, True],
                    [True, False, False],
                ]
            )
        ).all()

    # Returns a tensor of all False values when given a tensor of shape [batch_size] with all values equal to -1.
    def test_returns_attention_mask_with_all_false_values_negative_input(self):
        last_non_pad_token_per_batch = t.tensor([-1, -1, -1])
        model = SpeculativeDecodingWrapper()

        with pytest.raises(ValueError):
            attention_mask = model.get_attention_mask(last_non_pad_token_per_batch)


class TestGetKLastTokens:
    # Returns the last K tokens for each batch in the input tensor.
    def test_returns_last_k_tokens(self):
        input_tensor = t.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        last_non_pad_token_per_batch = t.tensor([3, 4])
        K = 2

        print(input_tensor)

        sd_model = SpeculativeDecodingWrapper()
        output = sd_model.get_k_last_tokens(
            input_tensor, last_non_pad_token_per_batch, K
        )

        expected_output = t.tensor([[3, 4], [9, 10]])

        assert output.shape == (2, 2)

        print(output)
        print(expected_output)

        assert t.all(output == expected_output)

    # Works with different K values.
    def test_works_with_different_k_values(self):
        input_tensor = t.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        last_non_pad_token_per_batch = t.tensor([3, 4])
        K = 3

        sd_model = SpeculativeDecodingWrapper()
        output = sd_model.get_k_last_tokens(
            input_tensor, last_non_pad_token_per_batch, K
        )

        expected_output = t.tensor([[2, 3, 4], [8, 9, 10]])
        assert t.all(output == expected_output)

    # Returns exception if K is 0.
    def test_returns_empty_tensor_if_k_is_zero(self):
        input_tensor = t.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        last_non_pad_token_per_batch = t.tensor([3, 4])
        K = 0

        sd_model = SpeculativeDecodingWrapper()
        with pytest.raises(AssertionError):
            output = sd_model.get_k_last_tokens(
                input_tensor, last_non_pad_token_per_batch, K
            )

    # Raises error if K is greater than sequence length.
    def test_returns_input_tensor_if_k_greater_than_or_equal_to_sequence_length(self):
        input_tensor = t.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        last_non_pad_token_per_batch = t.tensor([3, 4])
        K = 5

        sd_model = SpeculativeDecodingWrapper()
        with pytest.raises(RuntimeError):
            output = sd_model.get_k_last_tokens(
                input_tensor, last_non_pad_token_per_batch, K
            )

    def test_3d_tensor(self):
        input_tensor = t.tensor(
            [
                [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                [[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]],
            ]
        )
        last_non_pad_token_per_batch = t.tensor([3, 4])
        K = 2

        print("input_tensor.shape", input_tensor.shape)

        sd_model = SpeculativeDecodingWrapper()
        output = sd_model.get_k_last_tokens(
            input_tensor, last_non_pad_token_per_batch, K
        )
        assert output.shape == (2, 2, 2)

        print(output)

        assert t.all(
            output
            == t.tensor(
                [[[5, 6], [7, 8]], [[17, 18], [19, 20]]],
            )
        )
