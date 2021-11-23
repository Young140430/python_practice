import torch
from torch import nn

from data import MNISTDataset
from torch.utils.data import DataLoader
from net import net_v1
from torch import optim

DEVICE = "cuda"

class Trainer:
    def __init__(self,root):
        #加载训练数据
        self.train_dataset = MNISTDataset(root,True)
        self.train_loader = DataLoader(self.train_dataset,batch_size=512,shuffle=True)
        #创建模型
        self.net = net_v1()
        #将模型加载到GPU上
        self.net.to(DEVICE)
        #创建优化器
        self.opt = optim.Adam(self.net.parameters())
        #损失函数(均方差)
        self.loss_func = nn.MSELoss()

    #训练代码
    def __call__(self):
        for epoch in range(100000):
            for i,(imgs,tags) in enumerate(self.train_loader):
                #将数据加载到GPU上
                imgs,tags = imgs.to(DEVICE),tags.to(DEVICE)

                y = self.net.forward(imgs)
                loss = self.loss_func(y,tags)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                print(loss)
if __name__ == '__main__':
    trainer = Trainer("D:\MNIST_IMG")
    trainer()