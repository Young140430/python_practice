import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,24,7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24,52,7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(52,128,7),
            nn.ReLU(),
            nn.Conv2d(128, 256, 7),
            nn.ReLU(),
            nn.Conv2d(256, 512, 7),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3,padding=1),
            nn.ReLU()
        )

        self.out_layer = nn.Sequential(
            nn.Linear(1024*2*2,2),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        conv_out = self.layers(x)
        conv_out = conv_out.reshape(-1,1024*2*2)
        out = self.out_layer(conv_out)
        return out

if __name__ == '__main__':
    net = Net()
    x = torch.randn(1,3,100,100)
    y = net.forward(x)
    print(y.shape)