import numpy as np
from PIL import Image

#将图像转为矩阵
img = Image.open("../img/pic2.jpg")

img_data = np.array(img,dtype=np.uint8)
print(img_data)
print(img.size)
print(img_data.shape)
print(img_data.dtype)

img = Image.fromarray(img_data,"RGB")
img.show()

# img2 = img.copy()
# img2.show()