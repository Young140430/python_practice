import torch.optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda"

train_data = datasets.CIFAR10("D:\CIFAR10",train=True,transform=transforms.ToTensor(),download=True)
test_data = datasets.CIFAR10("D:\CIFAR10",train=False,transform=transforms.ToTensor(),download=True)

train_loader = DataLoader(train_data,batch_size=512,shuffle=True)
test_loader = DataLoader(test_data,batch_size=100,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(32*32*3,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        return self.fc_layer(x)

if __name__ == '__main__':
    summaryWriter = SummaryWriter("log")
    net = Net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()
    train_step = 0
    test_step = 0
    for epoch in range(100000):
        train_sum_loss = 0
        train_sum_acc = 0
        for i,(img,label) in enumerate(train_loader):
            img,label = img.to(DEVICE),label.to(DEVICE)
            img = img.reshape(-1,32*32*3)
            out = net(img)
            label = one_hot(label,10).float()
            loss = loss_func(out,label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            acc = torch.mean(torch.eq(torch.argmax(out,dim=1),torch.argmax(label,dim=1)).float())
            train_sum_acc = train_sum_acc + acc
            train_sum_loss = train_sum_loss + loss
            if i%10 ==0  and i!=0:
                train_loss = train_sum_loss / 10
                train_acc = train_sum_acc / 10
                summaryWriter.add_scalar("train_acc", train_acc, train_step)
                summaryWriter.add_scalar("train_loss", train_loss, train_step)
                print("train_loss:",train_loss.item())
                print("train_acc:",train_acc.item())
                train_sum_loss = 0
                train_sum_acc = 0
                train_step +=1
        test_sum_loss = 0
        test_sum_score = 0
        score = 0
        for i,(img,label) in enumerate(test_loader):
            img,label = img.to(DEVICE),label.to(DEVICE)
            img = img.reshape(-1,32*32*3)
            out = net(img)
            label = one_hot(label,10).float()
            loss = loss_func(out,label)
            score = torch.mean(torch.eq(torch.argmax(out,dim=1),torch.argmax(label,dim=1)).float())
            test_sum_score = test_sum_score + score
            test_sum_loss = test_sum_loss + loss
            if i % 10 == 0 and i != 0:
                test_loss = test_sum_loss / 10
                test_score = test_sum_score / 10
                summaryWriter.add_scalar("test_score", test_score, test_step)
                summaryWriter.add_scalar("test_loss", test_loss, test_step)
                print("test_loss:",test_loss.item())
                print("test_score:",test_score.item())
                test_sum_loss = 0
                test_sum_acc = 0
                test_step +=1
