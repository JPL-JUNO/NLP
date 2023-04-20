"""
@Description: mini-batch 版的全连接层变换
@Author(s): Stephen CUI
@Time: 2023-04-19 21:39:37
"""

import numpy as np
from numpy import ndarray


W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
x = np.random.randn(10, 2)
assert x.shape[1] == W1.shape[0]
h = np.dot(x, W1)


def sigmoid(x:ndarray)->ndarray:
    return 1 / (1 + np.exp(-x))
a = sigmoid(h)

W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

s = np.dot(a, W2) + b2