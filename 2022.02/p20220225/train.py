import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import UNet
import MKDataset
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

path = r"E:\baiduwangpan\BaiduNetdiskDownload\DUTS\DUTS-TR\DUTS-TR"
module = r"E:\unet_param\best_param.pt"
img_save_path = r"F:\unet_img"
epoch = 1

summaryWriter = SummaryWriter("logs")

net = UNet.MainNet().cuda()
optimizer = torch.optim.Adam(net.parameters())
loss_func = nn.BCELoss()

dataloader = DataLoader(MKDataset.MKDataset(path), batch_size=3, shuffle=True)

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
else:
    print('No Params!')
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

best_loss = 10
step = 0
while True:
    for i, (xs, ys) in enumerate(dataloader):
        xs = xs.cuda()
        ys = ys.cuda()
        xs_ = net(xs)

        loss = loss_func(xs_, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            # txt_word = f'epoch: {epoch},  count: {i},  loss: {loss}\n'
            print(f'epoch: {epoch},  count: {i},  loss: {loss}')

            f = open('loss.txt', 'a')
            f.write(f'epoch: {epoch},  count: {i},  loss: {loss}\n')
            f.close()


        # torch.save(net.state_dict(), module)
        # print('module is saved !')
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(net.state_dict(), r"E:\unet_param\best_param.pt")
            b_param = "Best_param saved successfully！--->"+str(best_loss)+"\n"
            print(b_param)
            fi = open('best_param.txt','a')
            fi.write(b_param)
            fi.close()

        if i % 350 == 0 and i > 0:
            torch.save(net.state_dict(), r"E:\unet_param\last_param.pt")
            print("Last_param saved successfully！")

        x = xs[0]
        x_ = xs_[0]
        y = ys[0]
        # print(y.shape)
        img = torch.stack([x,x_,y],0)
        # print(img.shape)

        save_image(img.cpu(), os.path.join(img_save_path,'{}.png'.format(i)))
        print("Img saved successfully !")
        summaryWriter.add_scalar("train_loss: ", loss.item(),step)
        step += 1
    epoch += 1