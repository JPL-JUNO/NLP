"""
@Description:
@Author(s): Stephen CUI
@Time: 2023-04-20 16:17:40
"""

import torch
import numpy as np

cnt = 0


def describe(x):
    global cnt
    print('----- {} times output -----'.format(cnt))
    cnt += 1
    print('Type: {}'.format(x.type()))
    print('Shape/Size: {}'.format(x.shape))
    print('Values: \n{}'.format(x))


describe(torch.Tensor(2, 3))
describe(torch.rand(2, 3))
describe(torch.randn(2, 3))

describe(torch.zeros(2, 3))
x = torch.ones(2, 3)
x.fill_(5)
describe(x)


x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
describe(x)


npy = np.random.rand(2, 3)
describe(torch.from_numpy(npy))

x = torch.FloatTensor([[1, 2, 3],
                       [4, 5, 6]])
describe(x)
x = x.long()
describe(x)

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.int64)
describe(x)

x = x.float()
describe(x)

x = torch.randn(2, 3)
describe(torch.add(x, x))

x = torch.arange(6)
describe(x)

x = x.view(2, 3)
describe(x)


describe(torch.sum(x, dim=0))
describe(torch.sum(x, dim=1))

describe(torch.transpose(x, 0, 1))

# indexing, slicing, and joining
x = torch.arange(6).view(2, 3)
describe(x)

describe(x[:1, :2])

describe(x[0, 1])


indices = torch.LongTensor([0, 2])
describe(torch.index_select(x, dim=1, index=indices))

indices = torch.LongTensor([0, 0])
# return 的结果与 indices 的长度相同，与 ndarray 是一样的
describe(torch.index_select(x, dim=0, index=indices))

row_indices = torch.arange(2).long()
col_indices = torch.LongTensor([0, 1])
# row_indices 与 col_indices 的长度要匹配，不匹配的话似乎会进行广播机制
describe(x[row_indices, col_indices])
