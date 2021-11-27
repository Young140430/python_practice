import torch
from torch import nn,optim
from torch.nn.functional import one_hot
from Conv2d_data import MyDataset
from torch.utils.data import DataLoader

DEVICE = "cuda"

class Conv_net(nn.Module):
    def __init__(self):
        super(Conv_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 24, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 52, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(52, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(1024*4*4,2),
            nn.Softmax(dim = 1)
        )

    def forward(self,x):
        conv_out = self.layer1(x)
        #形状变换
        conv_out = conv_out.reshape(-1,1024*4*4)
        out = self.layer2(conv_out)
        return out
        # return conv_out



if __name__ == '__main__':
    net = Conv_net()
    x = torch.randn(1, 3,100,100)
    y = net(x)
    print(y.shape)

