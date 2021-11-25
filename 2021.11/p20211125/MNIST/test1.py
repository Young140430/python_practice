import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter
DEVICE = "cuda"
summaryWriter=SummaryWriter("logs")
train_data = datasets.MNIST("E:\MNIST_data",train=True,transform=transforms.ToTensor(),download=False)
train_loader = DataLoader(train_data,batch_size=500,shuffle=True)
test_data = datasets.MNIST("E:\MNIST_data",train=False,transform=transforms.ToTensor(),download=False)
test_loader = DataLoader(test_data,batch_size=100,shuffle=True)
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
        conv_out = conv_out.reshape(-1,256*1*1)
        out = self.out_layer(conv_out)
        return out

if __name__ == '__main__':
    net = Net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()
    train_step = 0
    test_step = 0
    for epoch in range(100000):
        train_sum_loss = 0
        train_sum_acc = 0
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            out = net(img)
            label = one_hot(label, 10).float()
            train_loss = loss_func(out, label)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            train_sum_loss = train_sum_loss + train_loss.cpu().item()
            if i % 10 == 0 and i != 0:
                avg_train_loss = train_sum_loss / 10
                print(f"epoch==>{epoch}", f"i=={i}", "train_loss==>", avg_train_loss)
                # 收集训练损失
                summaryWriter.add_scalar("train_loss", avg_train_loss, train_step)
                train_step += 1
                train_sum_loss = 0
        score = 0
        for i, (img, label) in enumerate(test_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            out = net(img)
            label = one_hot(label, 10).float()
            test_loss = loss_func(out, label)
            print(f"epoch==>{epoch}", f"i=={i}", "test_loss==>", test_loss)
            summaryWriter.add_scalar("test_loss", test_loss, test_step)
            test_step += 1
