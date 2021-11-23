import torch
from torch import nn

class net_v1(nn.Module):
    #初始化网络结构组件
    def __init__(self):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(784,100),
            nn.ReLU(),
            nn.Linear(100,52),
            nn.ReLU(),
            nn.Linear(52,10),
            nn.Softmax(dim=1)
        )
    #网络前向计算
    def forward(self,x):
        return self.fc_layer(x)

if __name__ == '__main__':
    net = net_v1()
    x = torch.randn(1,784)
    y = net.forward(x)
    print(y.shape)