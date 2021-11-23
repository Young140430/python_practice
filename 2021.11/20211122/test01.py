from torchvision import datasets,transforms
from PIL import Image
import numpy as np


train_data = datasets.MNIST("MNIST",train=True,download=True)
test_data = datasets.MNIST("MNIST",train=False,download=True)

print(train_data)
print(test_data)
print(type(train_data))
#获取数据
print(train_data.data)
#获取标签
print(train_data.targets)

print(train_data.data.shape)
print(train_data.targets.shape)

print(train_data.data[0])
print(train_data.targets[0])

# img = Image.fromarray(np.array(train_data.data[0]),"L")
# img.show()
#使用transforms类实现张量与图像的转化
img_data = train_data.data[0]

# img_data = img_data/255.
print(img_data)
print(type(img_data))
#
# img_data_ = img_data * 255
# print(img_data_)
# img = Image.fromarray(np.array(img_data,dtype=np.uint8),"L")
# img.show()

unloader = transforms.ToPILImage()
img = unloader(img_data)
print(type(img))
img.show()