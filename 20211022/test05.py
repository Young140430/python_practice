import torch

a = torch.tensor([[[1,2,2],[3,4,3],[2,1,2],[3,1,3]]])
print(a)
print(a.shape)

b = torch.tensor([[[3,1,2],[3,1,3],[2,2,2],[3,3,3]]])
print(b)
print(b.shape)

c = torch.tensor([[[1,2],[3,4],[5,6]]])
print(c)
print(c.shape)

print(a-b)
# print(a+c)
print(a*b)


print(a*3)
#张量乘法
d = torch.matmul(a,c)
# d = a@c
print(d.shape)
