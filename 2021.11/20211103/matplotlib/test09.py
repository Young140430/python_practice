import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(20)
y = np.random.randn(20)

plt.scatter(x,y,s=200,label="boy",c="blue",marker=".")

#添加新数据
x = np.random.randn(10)
y = np.random.randn(10)
plt.scatter(x,y,s=150,label="girl",c="red",marker="+")
#显示图例
plt.legend()
plt.show()