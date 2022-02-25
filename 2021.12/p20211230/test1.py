import thop
from torchvision import models
import torch

net=models.densenet121()
input=torch.randn(1,3,256,256)
cs,params=thop.profile(net,(input,))
print(cs)
print(params)