from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image

tf=transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor()
])
class handle_data(Dataset):
    def __init__(self,root,is_tarin=True):
        self.datalist=[]
        sub_dir="train" if is_tarin else "test"
        path=fr"{root}\{sub_dir}"
        for man in os.listdir(path):
            for img_filename in os.listdir(os.path.join(path,man)):
                self.datalist.append([os.path.join(path,man,img_filename),int(man)])
    def __len__(self):
        return len(self.datalist)
    def __getitem__(self, index):
        img_data=tf(Image.open(self.datalist[index][0]))
        return img_data,self.datalist[index][1]





