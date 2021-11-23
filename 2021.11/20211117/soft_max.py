import torch

def soft_max(t,dim=0):
    h = torch.exp(t)
    z = torch.sum(h,dim=dim,keepdim=True)
    return h/z

# a = torch.Tensor([1,2,3])
a = torch.Tensor([[1,2,3]])
b = soft_max(a,dim=1)
print(b)
print(torch.sum(b))