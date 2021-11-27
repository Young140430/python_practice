from torch.utils.data import DataLoader,Dataset
from torch import nn
import torch
from Conv2d_data import MyDataset
from Conv2d_net import Conv_net
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import one_hot
import numpy as np

train_data = MyDataset("D:\cat_dog_img",is_train = True)
train_loader = DataLoader(train_data, 100, shuffle=True)

test_data = MyDataset("D:\cat_dog_img",is_train = False)
test_loader = DataLoader(test_data, 100, shuffle=True)

DEVICE = "cuda"
summaryWriter = SummaryWriter("logs")

if __name__ == '__main__':
    net = Conv_net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.MSELoss()
    step = 0
    test_step = 0

    for epoch in range(100000):
        sum_loss = 0
        sum_acc = 0
        for i,(img,target) in enumerate(train_loader):
            img,target = img.to(DEVICE),target.to(DEVICE)
            img = img.reshape(-1,3,100,100)
            out = net(img)


            loss = loss_func(out,target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            acc = torch.mean(torch.eq(torch.argmax(out, dim = 1), torch.argmax(target, dim = 1)).float())

            sum_acc = sum_acc + acc

            sum_loss = sum_loss + loss.cpu().item()

            if i%10==0 and i!=0:

                avg_loss = sum_loss / 10
                avg_acc = sum_acc/10

                summaryWriter.add_scalar("avg_train_loss",avg_loss,step)
                summaryWriter.add_scalar("avg_train_acc", avg_acc, step)


                print("loss==>",avg_loss)
                print("acc==>",acc)
                step+=1
                sum_loss = 0
                sum_acc = 0
        # 验证代码/test

        test_acc = 0
        test_loss = 0
        for i, (img, label) in enumerate(test_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)


            out = net(img)


            test_loss = loss_func(out, label)

            test_acc = torch.mean(torch.eq(torch.argmax(out, dim = 1), torch.argmax(label, dim = 1)).float())

            summaryWriter.add_scalar("test_acc", test_acc, test_step)
            summaryWriter.add_scalar("test_loss", test_loss, test_step)
            print("test_loss:", test_loss.item())
            print("test_acc:", test_acc.item())

            test_step += 1




