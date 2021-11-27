import torch
from torch.utils.data import Dataset
import os,cv2
import numpy as np

class MyDataset(Dataset):
    def __init__(self, root, is_train = True):
        """初始化数据集"""
        self.dataset = []
        sub_dir = "TRAIN" if is_train else "TEST"
        for tag in os.listdir(f"{root}/{sub_dir}"):
            img_dir = f"{root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"
                self.dataset.append((img_path, tag))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file_name = self.dataset[index]
        img = cv2.imread(file_name[0])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # print(img.shape)
        img = np.transpose(img,(2,0,1))
        # print(img.shape)
        #归一化
        img = np.array(img)/255.
        #将图片数据转为张量
        img = torch.tensor(img,dtype=torch.float32)

        target = file_name[1]
        target = torch.tensor([int(target)],dtype=torch.float32)

        tag_one_hot = np.zeros(2)
        tag_one_hot[int(target)] = 1
        return np.float32(img),np.float32(tag_one_hot)

if __name__ == '__main__':
    dataset = MyDataset("D:\cat_dog_img",is_train = True)
    print(dataset[0])