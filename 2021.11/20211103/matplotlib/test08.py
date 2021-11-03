import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# #创建测试数据
# x = np.random.randn(20)
# y = np.random.randn(20)
#
# #绘制点
# plt.scatter(x,y)
# plt.show()
# #绘制折线图
# plt.plot(x,y)
# plt.show()

# img = Image.open("../img/pic2.jpg")
# plt.imshow(img)
# plt.axis(False)
# plt.show()

# #实时绘图
# ax = []
# ay = []
# #开启绘画
# plt.ion()
# for i in range(100):
#     ax.append(i)
#     ay.append(i**2)
#     #清屏
#     plt.clf()
#     plt.plot(ax,ay)
#     plt.pause(0.01)
#
# #结束绘画
# plt.ioff()
# plt.show()

# x = np.random.normal(0,1,100)
# y = np.random.normal(0,1,100)
# z = np.random.normal(0,1,100)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x,y,z)
# plt.show()