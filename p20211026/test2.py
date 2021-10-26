import torch
import matplotlib.pyplot as plt
from torch import nn
import random
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

xs=torch.unsqueeze(torch.arange(-10,10),dim=1)/10
ys=[np.sin(e)*random.randint(1,6) for e in xs]
ys=torch.stack(ys)

class Line(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layer=nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,x):
        return self.fc_layer(x)
if __name__ == '__main__':
    net=Line()
    loss=nn.MSELoss()
    opt=torch.optim.Adam(net.parameters())

    plt.ion()
    for i in range(3000000):
        out=net.forward(xs)
        l=loss(out,ys)

        opt.zero_grad()
        l.backward()
        opt.step()

        if i%10==0:
            print(l.item())
            plt.cla()
            plt.title("loss%.4f" % l.item())
            plt.plot(xs, ys, ".")
            plt.plot(xs, out.detach())
            plt.pause(0.001)
    plt.ioff()