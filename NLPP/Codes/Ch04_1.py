"""
@Description: Multilayer perceptron using PyTorch
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-04-28 11:04:38
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """MLP 初始化

        Args:
            input_dim (int): the size of the input vectors
            hidden_dim (int): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
        """
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in: Tensor, apply_softmax: bool = False) -> Tensor:
        """The forward pass of the MLP

        Args:
            x_in (Tensor): an input data Tensor, x_in.shape should be (batch, input_dim)
            apply_softmax (bool, optional): a flag fot the softmax activation. should be false in used
            with the cross-entropy losses. Defaults to False.

        Returns:
            Tensor: the resulting tensor, tensor.shape should be (batch, output_dim)
        """
        intermediate = F.relu(self.fc1(x_in))
        output = self.fc2(intermediate)

        if apply_softmax:
            output = F.softmax(output, dim=1)
        return output


batch_size = 2
input_dim = 3
hidden_dim = 100
output_dim = 4
mlp = MultilayerPerceptron(
    input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
print(mlp)


def describe(x: Tensor) -> None:
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: {}".format(x))


x_input = torch.rand(batch_size, input_dim)
describe(x_input)
y_output = mlp(x_input, apply_softmax=False)
describe(y_output)
y_output = mlp(x_input, apply_softmax=True)
describe(y_output)
