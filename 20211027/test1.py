import torch
#print(torch.cuda.device_count())
cuda0 = torch.device('cuda:0')
x = torch.tensor([1., 2.], device=cuda0)
# x.device is device(type='cuda', index=0)
y = torch.tensor([1., 2.]).cuda()
# y.device is device(type='cuda', index=0)
# allocates a tensor on GPU 1
a = torch.tensor([1., 2.], device=cuda0)
# transfers a tensor from CPU to GPU 1
b = torch.tensor([1., 2.]).cuda()
# a.device and b.device are device(type='cuda', index=0)
# You can also use ``Tensor.to`` to transfer a tensor:
b2 = torch.tensor([1., 2.]).to(device=cuda0)
# b.device and b2.device are device(type='cuda', index=0)
c = a + b
# c.device is device(type='cuda', index=0)
z = x + y
# z.device is device(type='cuda', index=0)
# even within a context, you can specify the device
# (or give a GPU index to the .cuda call)
d = torch.randn(2, device=cuda0)
e = torch.randn(2).to(cuda0)
f = torch.randn(2).cuda(cuda0)
# d.device, e.device, and f.device are all device(type='cuda', index=0)
print(a)
print(b)
print(b2)
print(c)
print(z)
print(d)
print(e)
print(f)