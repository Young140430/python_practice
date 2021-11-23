import torch.optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda"

summaryWriter = SummaryWriter("logs")

train_data = datasets.MNIST("D:\data\MNIST_data",train=True,transform=transforms.ToTensor(),download=False)
test_data = datasets.MNIST("D:\data\MNIST_data",train=False,transform=transforms.ToTensor(),download=False)

train_loader = DataLoader(train_data,batch_size=512,shuffle=True)
test_loader = DataLoader(test_data,batch_size=512,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        return self.fc_layer(x)

if __name__ == '__main__':
    net = Net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()

    for epoch in range(1000):
        for i,(img,label) in enumerate(train_loader):
            img,label = img.to(DEVICE),label.to(DEVICE)
            #N,28,28==>N,28*28
            img = img.reshape(-1,28*28)
            # out = net.forward(img)
            out = net(img)
            #对标签做one_hot
            label = one_hot(label,10).float()
            loss = loss_func(out,label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i%10==0 and i!=0:
                print("loss:",loss.item())
                summaryWriter.add_scalar("train_loss",loss)
                acc = torch.mean(torch.eq(torch.argmax(out,dim=1),torch.argmax(label,dim=1)).float())
                _out = torch.argmax(out[:10],dim=1)
                _label = torch.argmax(label[:10],dim=1)
                print("acc:",acc)
                print("out_put",_out)
                print("label",_label)

