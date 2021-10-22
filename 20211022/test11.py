import torch
import matplotlib.pyplot as plt
from torch import nn

xs = torch.arange(0.01,1,0.01)
ys = 3*xs+4 + torch.rand(99)

class Line(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(1))
        self.b = torch.nn.Parameter(torch.rand(1))
    #前向计算
    def forward(self,x):
        return x * self.w + self.b
if __name__ == '__main__':
    line = Line()
    #定义损失函数(均方差)
    loss_func = nn.MSELoss()
    #梯度下降优化器
    opt = torch.optim.SGD(line.parameters(),lr=0.1)

    plt.ion()
    for epoch in range(30):
        for _x,_y in zip(xs,ys):
            z = line.forward(_x)
            loss = loss_func(z,_y)

            #清空梯度
            opt.zero_grad()
            #自动求导
            loss.backward()
            #更新梯度
            opt.step()
            #tensor.item()  将张量转为标量
            print(line.w.item(),line.b.item())

            plt.cla()
            plt.plot(xs,ys,".")
            v = [line.w.detach() * e + line.b.detach() for e in xs]
            plt.plot(xs,v)
            plt.pause(0.001)
    plt.ioff()
    plt.show()