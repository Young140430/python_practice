import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import os
import gzip
import numpy as np
from gevent.testing import params
from paddle.optimizer import lr
from torch import nn
from torch.nn import init
from torch.nn.functional import cross_entropy

sys.path.append("..")
#调用库时，sys.path会自动搜索路径，为了导入d2l这个库，所以需要添加".."
from IPython import display
#在这一节d2l库仅仅在绘图时被使用，因此使用这个库做替代
#获取训练集
def load_data(folder_path):
    files = ['y_train.gz', 'x_train.gz', 'y_test.gz', 'x_test.gz']
    paths = [os.path.join(folder_path, f) for f in files]
    with gzip.open(paths[0],'rb') as f1:
        y_train = np.frombuffer(f1.read(),np.uint8,offset=8)
    with gzip.open(paths[1],'rb') as f2:
        X_train = np.frombuffer(f2.read(),np.uint8,offset=16).reshape(len(y_train),28,28)
    with gzip.open(paths[2],'rb') as f3:
        y_test = np.frombuffer(f3.read(),np.uint8,offset=8)
    with gzip.open(paths[3],'rb') as f4:
        X_test = np.frombuffer(f4.read(),np.uint8,offset=16).reshape(len(y_test),28,28)
    return (X_train,y_train),(X_test,y_test)
(X_train ,y_train),(X_test ,y_test) = load_data(r"E:/dataset")
mnist_train=[]
mnist_test=[]
for i in range(len(X_train)):
    mnist_train.append([torch.tensor(X_train[i]),torch.tensor(y_train[i])])
#获取测试集
for j in range(len(X_test)):
    mnist_test.append([torch.tensor(X_test[j]),torch.tensor(y_test[j])])
#print(len(X_train),len(y_train),len(X_test),len(y_test))	#输出训练集的样本数
#print(torch.tensor(X_train[0]))	#通过下标访问任意一个样本，返回值为两个torch，一个特征tensor和一个标签tensor
#print(torch.tensor(y_train[0]))
#print(mnist_train[0])
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
#labels是一个列表
#数值标签转文本标签
def show_fashion_mnist(images, labels):
    display.display_svg()#绘制矢量图
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    #创建子图，一行len(images)列，图片大小12*12
    for f, img, lbl in zip(figs, images, labels):
        #zip函数将他们压缩成由多个元组组成的列表
        f.imshow(img.view((28, 28)).numpy())
        #将img转形为28*28大小的张量，然后转换成numpy数组
        f.set_title(lbl)
        #设置每个子图的标题为标签
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        #关闭x轴y轴
    plt.show()
'''X,y = [],[]
#初始化两个列表
for i in range(5):
	X.append(mnist_train[i][0])
	#循环向X列表添加图像
	y.append(mnist_train[i][1])
	#循环向y列表添加标签
show_fashion_mnist(X,get_fashion_mnist_labels(y))
#显示图像和列表'''
batch_size = 256
#小批量数目
train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle = True,num_workers = 0)
#num_workers=0,不开启多线程读取。
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size = batch_size,shuffle=False,num_workers=0)
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        # 定义一个输入层

    # 定义向前传播（在这个两层网络中，它也是输出层）
    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        # 将x换形为y后，再继续向前传播
        return y


net = LinearNet(num_inputs, num_outputs)
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
def accuracy(y_hat,y):
    return (y_hat.argmax(dim=1).float().mean().item())
def net_accurary(data_iter,net):
    right_sum,n = 0.0,0
    for X,y in data_iter:
    #从迭代器data_iter中获取X和y
        right_sum += (net(X).argmax(dim=1)==y).float().sum().item()
        #计算准确判断的数量
        n +=y.shape[0]
        #通过shape[0]获取y的零维度（列）的元素数量
    return right_sum/n


num_epochs = 5


# 一共进行五个学习周期

def train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer, net_accurary):
    for epoch in range(num_epochs):
        # 损失值、正确数量、总数 初始化。
        train_l_sum, train_right_sum, n = 0.0, 0.0, 0

        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 数据集损失函数的值=每个样本的损失函数值的和。
            optimizer.zero_grad()  # 对优化函数梯度清零
            l.backward()  # 对损失函数求梯度
            optimizer(params, lr, batch_size)

            train_l_sum += l.item()
            train_right_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_acc = net_accurary(test_iter, net)  # 测试集的准确率
        print('第%d学习周期, 误差%.4f, 训练准确率%.3f, 测试准确率%.3f' % (epoch + 1, train_l_sum / n, train_right_sum / n, test_acc))

#train_softmax(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, optimizer, net_accurary)



