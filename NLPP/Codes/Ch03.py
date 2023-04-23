"""
@Description: Chapter 3: Foundational Components of Neural Networks
@Author(s): Stephen CUI
@LastEditor(s): somebody name
@CreatedTime: 2023-04-23 11:42:18
"""


import torch
import torch.nn as nn
from torch.optim import optim
from torch import Tensor
import numpy as np
from typing import Tuple


class Percetron(nn.Module):
    def __init__(self, input_dim: int) -> None:
        """初始化

        Args:
            input_dim (int): size of the input features
        """
        super(Percetron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in: Tensor)->Tensor:
        """The forward pass of the perceptron

        Args:
            x_in (Tensor): an input data tensor
                x_in.shape should be (batch, num_features)
        Returns:
            Tensor: tensor.shape should be (batch, )
        """
        return torch.sigmoid(self.fc1(x_in)).squeeze()

LEFT_CENTER = (3, 3)
RIGHT_CENTER = (3,-2)
def get_toy_data(batch_size:int, left_center:tuple=LEFT_CENTER,right_center:tuple=RIGHT_CENTER)->Tuple[Tensor, Tensor]:
    """随机生成一些数据

    Args:
        batch_size (int): batch size
        left_center (tuple, optional): 一类数据的中心点. Defaults to LEFT_CENTER.
        right_center (tuple, optional): 另一类数据的中心点. Defaults to RIGHT_CENTER.

    Returns:
        Tuple[Tensor, Tensor]: 返回生成数据的张量，分别为二维的X，和一维的y
    """
    x_data = []
    y_targets = np.zeros(batch_size)
    for batch_i in range(batch_size):
        if np.random.random() >.5:
            x_data.append(np.random.normal(loc=left_center))
        else:
            x_data.append(np.random.normal(loc=right_center))
            y_targets[batch_i] = 1
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_targets, dtype=torch.float32)

input_dim = 2
lr = .001
perceptron = Percetron(input_dim=input_dim)
bce_loss = nn.BCELoss()
optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)



batch_size=1000
n_epochs=12
n_batches = 5


for epoch_i in range(n_epochs):
    for batch_i in range(n_batches):
        x_data, y_target = get_toy_data(batch_size)
        # Step 1: Clear the gradients
        perceptron.zero_grad()
        # Step 2: Compute the forward pass of the model
        y_pred = perceptron(x_data, apply_sigmoid=True)
        # Step 3: Compute the loss value that we wish to optimize
        loss = bce_loss(y_pred, y_target)
        # Step 4: Propagate the loss signal backward
        loss.backward()
        # Step 5: Trigger the optimizer to perform one update
        optimizer.step()
        
        