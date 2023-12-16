import torch

from resnet18 import MyConv2d
from torch.nn import Conv2d

m1 = MyConv2d(16, 16, 3, stride=1, padding=1)
m2 = Conv2d(16, 16, 3, stride=1, padding=1)

m1.weight = m2.weight
m1.bias = m2.bias

input = torch.randn(20, 16, 50, 100)

o1 = m1(input)
o2 = m2(input)

assert o1.shape == o2.shape
assert torch.all(o1.isclose(o2, atol=1e-6))
