from typing import Optional

import torch as t
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
        self,
        dim_observation: int,
        num_actions: int,
        hidden_sizes: Optional[list[int]] = [120, 84],
        dropout: float = 0.2,
    ):
        super().__init__()
        assert hidden_sizes is not None, "Hidden size is None"

        l1 = nn.Linear(dim_observation, hidden_sizes[0])
        relu = nn.ReLU()
        l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        l3 = nn.Linear(hidden_sizes[1], num_actions)
        self.dropout = nn.Dropout(p=dropout)

        self.seq = nn.Sequential(l1, relu, self.dropout, l2, relu, l3)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Convert an observation to an action.

        Parameters
        ----------
        x : t.Tensor
            _description_

        Returns
        -------
        t.Tensor
            _description_
        """
        return self.seq(x)


if __name__ == "__main__":
    net = QNetwork(dim_observation=4, num_actions=2)
    n_params = sum((p.nelement() for p in net.parameters()))
    print(net)
    print(f"Total number of parameters: {n_params}")
    assert n_params == 10934
