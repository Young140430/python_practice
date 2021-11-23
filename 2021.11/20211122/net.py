import torch
from torch import nn


class net_v1(nn.Module):
    #初始化网络结构组件
    def __init__(self):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(10000,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        )
    #网络前向计算
    def forward(self,x):
        return self.fc_layer(x)


if __name__ == '__main__':
    net = net_v1()
    x = torch.randn(1,10000)
    y = net.forward(x)
    print(y.shape)