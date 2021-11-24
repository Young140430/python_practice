import torch
from torch import nn
#m=nn.MaxPool2d(3,stride=2)
m = nn.MaxPool2d((3, 2), stride=(2, 1))
input = torch.randn(20, 16, 50, 32)
output = m(input)
print(output.shape)