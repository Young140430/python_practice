import torch

a = torch.tensor([45])
print(a.dtype)
# a = torch.Tensor([45])
# print(a.dtype)

b = torch.softmax(a,dim=0)
print(b)
print(torch.sum(b))
