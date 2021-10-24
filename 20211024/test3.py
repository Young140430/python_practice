import torch
import matplotlib.pyplot as plt
from torch import nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

xs = torch.unsqueeze(torch.arange(0.01,1,0.01),dim=1)
ys = 5*xs+1


class Line(nn.Module):
    #设计神经网络
    def __init__(self):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(1, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 1)
        )
    #前向计算
    def forward(self,x):
        return self.fc_layer(x)
if __name__ == '__main__':
    net = Line()
    #定义损失函数(均方差)
    loss_func = nn.MSELoss()
    #梯度下降优化器
    # opt = torch.optim.SGD(net.parameters(),lr=0.0005)
    opt = torch.optim.Adam(net.parameters())

    plt.ion()
    for epoch in range(3000):
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
    plt.show()