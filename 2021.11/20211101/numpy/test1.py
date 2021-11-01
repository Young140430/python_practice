import numpy as np

print("使用numpy生成一维数组")
data = [1,2,3,4,6,8,10]
x = np.array(data)

print(data)
print(x)
print(type(data))
print(type(x))
#查看数组的元素类型
print(x.dtype)

print("使用列表生成二维数组")
data = [[1,2,4],[3,4,5],[5,6,8]]
print(data)
x = np.array(data)
print(x)
print(x.ndim)
print(x.shape)
print(x.size)

print("使用zero/ones/empty创建数组：通过shape来创建")
x = np.zeros((5,4),dtype=np.int8)
print(x)
print(x.dtype)

x = np.ones((5,4),dtype=np.float32)
print(x)
print(x.dtype)

x = np.empty((5,4))
print(x)

print("使用arange生成连续的元素")
print(np.arange(10))
print(np.arange(3,19,2))

print("复制数组，并转换类型")
x = np.array([1,12,3,14,5],dtype=np.float64)
y = x.astype(np.int8)
print(x)
print(y)