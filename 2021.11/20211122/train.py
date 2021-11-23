import torch
from torch import nn

from data import Cat_Dog_Dataset
from torch.utils.data import DataLoader
from net import net_v1
from torch import optim
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda"

class Trainer:
    def __init__(self,root):
        self.summaryWriter = SummaryWriter("logs2")

        #加载训练数据
        self.train_dataset = Cat_Dog_Dataset(root,True)
        self.train_loader = DataLoader(self.train_dataset,batch_size=512,shuffle=True)

        # 加载测试数据
        self.test_dataset = Cat_Dog_Dataset(root,is_train=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=100, shuffle=True)
        #创建模型
        self.net = net_v1()
        # #加载预训练参数
        # self.net.load_state_dict(torch.load("param/10.pt"))
        #将模型加载到GPU上
        self.net.to(DEVICE)
        #创建优化器
        self.opt = optim.Adam(self.net.parameters())
        #损失函数(均方差)
        self.loss_func = nn.MSELoss()


    def __call__(self):
        test_step = 0
        train_step = 0
        log_test_loss = 100
        for epoch in range(100000):
            train_sum_loss = 0
            # 训练代码
            for i,(imgs,tags) in enumerate(self.train_loader):
                #将数据加载到GPU上
                imgs,tags = imgs.to(DEVICE),tags.to(DEVICE)

                y = self.net.forward(imgs)
                train_loss = self.loss_func(y,tags)

                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()
                train_sum_loss = train_sum_loss + train_loss.cpu().item()
                # print("模型输出的结果：==》",y)
                # print("模型输出的结果：==》",torch.sum(y,dim=1))
                # print("模型标签的结果：==》", tags)
                if i%10==0 and i!=0:
                    avg_train_loss = train_sum_loss / 10
                    print(f"epoch==>{epoch}",f"i=={i}","train_loss==>", avg_train_loss)
                    # 收集训练损失
                    self.summaryWriter.add_scalar("train_loss",avg_train_loss,train_step)
                    train_step += 1
                    train_sum_loss = 0
            score = 0
            #验证代码
            for i,(imgs,tags) in enumerate(self.test_loader):
                # 将数据加载到GPU上
                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)
                y = self.net.forward(imgs)
                test_loss = self.loss_func(y, tags)

                # 将one_hot编码转回输原始标签
                out_tags = torch.argmax(tags, dim=1)
                out_y = torch.argmax(y, dim=1)
                #验证精度
                score = torch.mean(torch.eq(out_y,out_tags).float())

                print(f"epoch==>{epoch}",f"i=={i}","test_loss==>", test_loss.item())
                print(f"epoch==>{epoch}",f"i=={i}","score==>", score.item())
                # 收集验证损失
                self.summaryWriter.add_scalar("test_loss", test_loss, test_step)
                self.summaryWriter.add_scalar("score",score,test_step)
                test_step += 1

                if test_loss < log_test_loss:
                    # 保存模型的权重
                    #torch.save(self.net.state_dict(), f"param/mnist.pt")
                    #print("参数保存成功！")
                    log_test_loss = test_loss

            #保存模型的权重
            #torch.save(self.net.state_dict(),f"param/{epoch}.pt")
            #print("epoch参数保存成功！")
if __name__ == '__main__':
    trainer = Trainer("D:\cat_dog_img")
    trainer()