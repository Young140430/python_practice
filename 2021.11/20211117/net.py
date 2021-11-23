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

class net_v2(nn.Module):
    def __init__(self):
        super(net_v2, self).__init__()

        self.W = nn.Parameter(torch.randn(784,10))
        self.B = nn.Parameter(torch.zeros(10))


    def forward(self,x):
        h = x@self.W + self.B

        #SoftMax
        h = torch.exp(h)
        z = torch.sum(h,dim=1,keepdim=True)
        return h/z

class net_v3(nn.Module):
    def __init__(self):
        super(net_v3, self).__init__()
        self.fc1 = nn.Linear(784,10)

    def forward(self,x):
        return torch.softmax(self.fc1(x),dim=1)

class net_v4(nn.Module):
    #初始化网络结构组件
    def __init__(self):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24,10),
            nn.Softmax(dim=1)
        )
    #网络前向计算
    def forward(self,x):
        return self.fc_layer(x)
class net_v5(nn.Module):
    #初始化网络结构组件
    def __init__(self):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.Softmax(dim=1)
        )
    #网络前向计算
    def forward(self,x):
        return self.fc_layer(x)
if __name__ == '__main__':
    net = net_v4()
    x = torch.randn(1,784)
    y = net.forward(x)
    print(y.shape)