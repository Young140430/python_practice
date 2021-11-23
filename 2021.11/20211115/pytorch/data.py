#pytorch已经提供好了数据集的处理父类
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os,cv2

class MNISTDataset(Dataset):
    #初始化数据集
    def __init__(self,root,is_train=True):
        self.dataset = []#记录所有的数据
        sub_dir = "TRAIN" if is_train else "TEST"
        for tag in os.listdir(f"{root}/{sub_dir}"):
            img_dir = f"{root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"
                self.dataset.append((img_path,tag))

    #统计数据集的长度
    def __len__(self):
        return len(self.dataset)
    #每条数据的处理方式
    def __getitem__(self, indxe):
        data = self.dataset[indxe]

        img_data = cv2.imread(data[0],cv2.IMREAD_GRAYSCALE)
        img_data = img_data.reshape(-1)#将数据变为一维
        img_data = img_data/255#归一化

        #one_hot
        tag_one_hot = np.zeros(10)
        tag_one_hot[int(data[1])] = 1

        return np.float32(img_data),np.float32(tag_one_hot)

if __name__ == '__main__':
    dataset = MNISTDataset("D:\MNIST_IMG",is_train=True)
    train_loader = DataLoader(dataset,batch_size=500,shuffle=True)
    for i,(x,y) in enumerate(train_loader):
        print(i)
        print(x.shape)
        print(y.shape)