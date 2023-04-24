"""
@Description: Chapter 3: Foundational Components of Neural Networks
@Author(s): Stephen CUI
@LastEditor(s): somebody name
@CreatedTime: 2023-04-23 11:42:18
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


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
    return torch.tensor(np.stack(x_data), dtype=torch.float32), torch.tensor(np.stack(y_targets), dtype=torch.float32)

def visualize_results(perceptron, x_data, y_truth, n_samples=1000,
                      ax=None, epoch=None, title='', levels=[.3, .4, .5], linestyles=['--','-','--']):
    y_pred = perceptron(x_data)
    y_pred = (y_pred >.5).long().data.numpy().astype(np.int32)
    
    x_data = x_data.data.numpy()
    y_truth = y_truth.data.numpy().astype(np.int32)
    
    n_classes = 2
    all_x =[[] for _ in range(n_classes)]
    all_colors = [[] for _ in range(n_classes)]
    
    markers = ['o', '*']
    
    for x_i, y_pred_i, y_true_i in zip(x_data, y_pred, y_truth):
        all_x[y_true_i].append(x_i)
        if y_pred_i == y_true_i:
            all_colors[y_true_i].append('white')
        else:
            all_colors[y_true_i].append('black')
    all_x = [np.stack(x_list) for x_list in all_x]
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    for x_list, color_list, marker in zip(all_x, all_colors, markers):
        ax.scatter(x_list[:, 0], x_list[:, 1], edgecolor='black', marker=marker, facecolor=color_list, s=100)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    Z = perceptron(torch.tensor(xy, dtype=torch.float32)).detach().numpy().reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=levels, linestyles=linestyles)
    plt.suptitle(title)
    if epoch is not None:
        plt.text(xlim[0], ylim[1], 'Epoch = {}'.format(str(epoch)))
seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

batch_size = 1000
x_data, y_truth = get_toy_data(batch_size)
_, ax = plt.subplots(1, 1, figsize=(10, 4))

left_x = []
right_x = []
left_colors = []
right_colors = []

for x_i, y_true_i in zip(x_data, y_truth):
    color = 'black'
    
    if y_true_i ==0:
        left_x.append(x_i)
        left_colors.append(color)
    else:
        right_colors.append(color)
        right_x.append(x_i)
left_x = np.stack(left_x)
right_x = np.stack(right_x)

ax.scatter(left_x[:, 0], left_x[:, 1], 
           color=left_colors, marker='o', s=100, facecolor='white')
ax.scatter(right_x[:, 0], right_x[:, 1], 
           color=right_colors, 
           marker='*', s=100)
plt.axis('off')

input_dim = 2
lr = .01
perceptron = Percetron(input_dim=input_dim)
bce_loss = nn.BCELoss()
losses = []
optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)

x_data_static, y_truth_static = get_toy_data(batch_size)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
visualize_results(perceptron, x_data_static, y_truth_static, ax=ax, title='Initial Model State')
plt.axis('off')

n_epochs = 12 
n_batches = 5
change = 1.0
# 上一次的损失函数保存
last = 10.0
epsilon = 1e-3
epoch = 0

x_data, y_target = get_toy_data(batch_size)

while change > epsilon or epoch < n_epochs or last > .3:
    for _ in range(n_batches):
        optimizer.zero_grad()
        y_pred = perceptron(x_data).squeeze()
        loss = bce_loss(y_pred, y_target)
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        losses.append(loss_value)
        
        change = abs(last-loss_value)
        last = loss_value
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    visualize_results(perceptron, x_data_static, y_truth_static, ax=ax,
                      epoch=epoch, title='{}, {}'.format(loss_value, change))
    plt.axis('off')
    epoch += 1
# for epoch_i in range(n_epochs):
    # for batch_i in range(n_batches):
    #     x_data, y_target = get_toy_data(batch_size)
    #     # Step 1: Clear the gradients
    #     perceptron.zero_grad()
    #     # Step 2: Compute the forward pass of the model
    #     y_pred = perceptron(x_data, apply_sigmoid=True)
    #     # Step 3: Compute the loss value that we wish to optimize
    #     loss = bce_loss(y_pred, y_target)
    #     # Step 4: Propagate the loss signal backward
    #     loss.backward()
    #     # Step 5: Trigger the optimizer to perform one update
    #     optimizer.step()
        
        