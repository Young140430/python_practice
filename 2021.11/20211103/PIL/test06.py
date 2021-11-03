import numpy as np
from PIL import Image

img = Image.open("../img/pic.jpeg")
img = img.resize((430,448))
img_data = np.array(img)
print(img_data.shape)
#w,h 切分为[2,h/2,2,w/2,3]
img_data = img_data.reshape((2,448//2,2,430//2,3))
print("切分之后的形状：",img_data.shape)#(2, 224, 2, 215, 3)
#(2, 224, 2, 215, 3)==》（2，2，224，215，3）
img_data = img_data.transpose(0,2,1,3,4)
print("转置之后的形状：",img_data.shape)
#转置之后通过reshape重新设置形状，变为4张图像
img_data = img_data.reshape(-1,224,215,3)
print("转置之后通过reshape重新设置形状:",img_data.shape)
#从最外层切分，切分为4块，每一块是一个小图
img_arr = np.split(img_data,4,axis=0)
for i,img_id in enumerate(img_arr):
    img = Image.fromarray(img_id[0])
    img.save(f"../img/s{i}.jpg")