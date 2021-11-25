from net import Net
from data import MNIST_Dataset
import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    DEVICE="cuda"
    summaryWriter = SummaryWriter("logs1")
    dataset1 = MNIST_Dataset("E:\MNIST_IMG", is_train=True)
    train_loader = DataLoader(dataset1, batch_size=100, shuffle=True)
    #train_loader=train_loader.reshape(-1,1,28,28)
    dataset2 = MNIST_Dataset("E:\MNIST_IMG", is_train=False)
    test_loader = DataLoader(dataset2, batch_size=50, shuffle=True)
    #test_loader=test_loader.reshape(-1,1,28,28)
    net = Net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()
    train_step=0
    test_step=0
    for epoch in range(100000):
        train_sum_loss=0
        train_sum_acc=0
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            img=img.reshape(-1,1,28,28)
            out = net(img)
            train_loss = loss_func(out, label)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            train_sum_loss = train_sum_loss + train_loss.cpu().item()
            if i % 10 == 0 and i != 0:
                avg_train_loss=train_sum_loss/10
                print(f"epoch==>{epoch}", f"i=={i}", "train_loss==>", avg_train_loss)
                # 收集训练损失
                summaryWriter.add_scalar("train_loss", avg_train_loss, train_step)
                train_step += 1
                train_sum_loss = 0
        score=0
        for i, (img, label) in enumerate(test_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            img = img.reshape(-1, 1, 28, 28)
            out = net(img)
            test_loss = loss_func(out, label)
            print(f"epoch==>{epoch}", f"i=={i}", "test_loss==>", test_loss)
            summaryWriter.add_scalar("test_loss", test_loss, test_step)
            test_step += 1
