import os.path

from data import handle_data
from net import facenet
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, root):
        self.train_data = handle_data(root, True)
        self.train_loader = DataLoader(self.train_data, 60, True)
        self.test_data = handle_data(root, False)
        self.test_loader = DataLoader(self.test_data, 20, True)
        self.net = facenet().cuda()
        self.opt = optim.Adam(self.net.parameters())
        self.nllloss = nn.NLLLoss()
        self.summarywriter = SummaryWriter("log")

    def __call__(self):
        epoch = 1
        loss_min = 1000
        net_path="param/net_last.pt"
        if os.path.exists(net_path):
            print("成功加载已有参数!")
            self.net.load_state_dict(torch.load(net_path))
        while True:
            # train
            loss_sum = 0
            for xs, tags in self.train_loader:
                xs, tags = xs.cuda(), tags.cuda()
                cls = self.net(xs)

                loss = self.nllloss(cls, tags)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                loss_sum += loss.detach().cpu().item()
                del xs, tags, loss

            loss_avg = loss_sum / len(self.train_loader)
            print(f"epoch:{epoch}==>train_loss:{loss_avg}")
            self.summarywriter.add_scalar("train", loss_avg, epoch)

            # test
            loss_sum = 0
            for xs, tags in self.test_loader:
                xs, tags = xs.cuda(), tags.cuda()
                cls = self.net(xs)

                loss = self.nllloss(cls, tags)
                loss_sum += loss.cpu().item()
                del xs, tags, loss

            loss_avg = loss_sum / len(self.test_loader)
            print(f"epoch:{epoch}==>test_loss:{loss_avg}")
            self.summarywriter.add_scalar("test", loss_avg, epoch)
            if loss_avg < loss_min:
                loss_min = loss_avg
                torch.save(self.net.state_dict(), "param/net_best_new.pt")
                print(f"epoch:{epoch},参数保存成功!")
            else:
                print(f"test_loss大于最小值,不保存!")

            torch.save(self.net.state_dict(), f"param/net_last_new.pt")

            epoch += 1


if __name__ == '__main__':
    trainer = Trainer(r"F:\face_recognition")
    trainer()
