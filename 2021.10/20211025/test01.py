import torch
import matplotlib.pyplot as plt
from torch import nn
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

xs = torch.unsqueeze(torch.arange(-20.,20.),dim=1)/20
ys = [e.pow(3)*random.randint(1,6) for e in xs]
ys = torch.stack(ys)

class Line(nn.Module):
    #设计神经网络
    def __init__(self):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 240),
            nn.ReLU(),
            nn.Linear(240, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    #前向计算
    def forward(self,x):
        return self.fc_layer(x)
if __name__ == '__main__':
    net = Line()
    #定义损失函数(均方差)
    loss_func = nn.MSELoss()
    #梯度下降优化器
    # opt = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    opt = torch.optim.Adam(net.parameters())

    plt.ion()
    for epoch in range(3000000):
        out = net.forward(xs)

        loss = loss_func(out,ys)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 5 ==0:
            print(loss.item())

            plt.cla()
            plt.title("loss%.4f"%loss.item())
            plt.plot(xs,ys,".")
            plt.plot(xs,out.detach())
            plt.pause(0.001)
    plt.ioff()
    # plt.show()