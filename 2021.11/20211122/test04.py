import torch
from torch.nn.functional import one_hot

a = torch.Tensor([1,3,5]).long()
print(a)
y = label = one_hot(a,num_classes=10)
print(y)