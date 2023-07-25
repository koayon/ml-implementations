import torch as t


def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """
    batch = x.shape[0]
    in_channels = x.shape[1]

    left_pad = t.full((batch, in_channels, left), pad_value)
    right_pad = t.full((batch, in_channels, right), pad_value)

    return t.cat((left_pad, x, right_pad), dim=2)


def pad2d(
    x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float
) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    batch = x.shape[0]
    in_channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]

    top_pad = t.full((batch, in_channels, top, width), pad_value)
    bottom_pad = t.full((batch, in_channels, bottom, width), pad_value)
    left_pad = t.full((batch, in_channels, top + height + bottom, left), pad_value)
    right_pad = t.full((batch, in_channels, top + height + bottom, right), pad_value)

    middle = t.cat((top_pad, x, bottom_pad), dim=2)
    return t.cat((left_pad, middle, right_pad), dim=3)
