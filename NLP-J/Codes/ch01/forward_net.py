"""
@Description: forward net
@Author(s): Stephen CUI
@Time: 2023-04-19 22:09:03
"""

import numpy as np
from numpy import ndarray


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W: ndarray, b: ndarray) -> None:
        self.params = [W, b]

    def forward(self, x: ndarray) -> ndarray:
        W, b = self.params
        assert x.shape[1] == W.shape[0]
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        I, H, O = input_size, hidden_size, output_size
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        self.layers = [Affine(W1, b1),
                       Sigmoid(),
                       Affine(W2, b2)]
        self.params = []
        for layer in self.layers:
            self.params + + layer.params

    def predict(self, x: ndarray) -> ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
