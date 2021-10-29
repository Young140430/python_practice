import numpy as np
import torch

a = np.array([[[1,2],[3,4]]])
print(type(a))
print(a)
print(a.shape)
print(a.ndim)

b = torch.tensor([[[1,2],[3,4]]])
print(b)
print(b.shape)
