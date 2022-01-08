from net import facenet
from PIL import Image
from data import tf
from torch.nn.functional import normalize
import torch,os

path=r"F:\face_recognition\database"
net=facenet()
net.load_state_dict(torch.load("param/net_last_new.pt"))
net.eval()
man_list=[]
for i in os.listdir(path):
    pic_list = torch.Tensor([])
    for filename in os.listdir(fr"{path}\{i}"):
        data=tf(Image.open(fr"{path}\{i}\{filename}"))
        feat=net.encode(data[None,...])
        pic_list=torch.cat((pic_list,feat))
        print(f"{i} {filename} done!")
    man_list.append(pic_list)
man_list=torch.stack(man_list)
print(man_list)
torch.save(man_list,fr"F:\face_database\all_feat_new.pt")


