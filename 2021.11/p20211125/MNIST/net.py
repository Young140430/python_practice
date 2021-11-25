import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,24,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24,52,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(52,128,3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3,padding=1),
            nn.ReLU()
        )

        self.out_layer = nn.Sequential(
            nn.Linear(256*1*1,10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        conv_out = self.layers(x)
        #1, 256, 1, 1 ==>1,256*1*1
        conv_out = conv_out.reshape(-1,256*1*1)
        out = self.out_layer(conv_out)
        return out

if __name__ == '__main__':
    net = Net()
    x = torch.randn(1,3,28,28)
    y = net.forward(x)
    print(y.shape)