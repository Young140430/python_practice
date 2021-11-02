import numpy as np

print("使用numpy生成一维数组")
data = [1,2,3,4]
x = np.array(data)

print(data)
print(x)
print(type(data))
print(type(x))
#查看数组的元素类型
print(x.dtype)

print("使用列表生成二维数组")
data = [[1,2],[3,4],[5,6]]
print(data)
x = np.array(data)
print(x)
print(x.ndim)
print(x.shape)
print(x.size)

print("使用zero/ones/empty创建数组：通过shape来创建")
x = np.zeros((3,4),dtype=np.int8)
print(x)
print(x.dtype)

x = np.ones((3,4),dtype=np.float32)
print(x)
print(x.dtype)

x = np.empty((3,4))
print(x)

print("使用arange生成连续的元素")
print(np.arange(6))
print(np.arange(3,10,2))

print("复制数组，并转换类型")
x = np.array([1,2,3,4,5],dtype=np.float64)
y = x.astype(np.int8)
print(x)
print(y)

print("将字符串元素转为数值元素")

x = np.array(["1","2","3","4","5"],dtype=np.string_)
y = x.astype(dtype=np.float32)
print(x)
print(y)

print("使用其他数组的数据类型作为参数")
x = np.array([1,2,3,4],dtype=np.float64)
y = np.arange(3,dtype=np.int8)
print(y)
print(y.astype(x.dtype))

print("ndarray的数组与标量/数组的运算")
x = np.array([1,2,3])
print(x*2)
print(x>2)
y = np.array([3,4,5])
print(x*y)
print(x>y)

print("ndarray的基本索引")
x = np.array([[1,2],[3,4],[5,6]])
print(x)
print(x[1])
print(x[1][1])
print(x[1,1])

x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(x[1,0])
y = x[1,0].copy()#生成一个副本
# z = x[1,0]#未生成副本

# z[0] = 0
y[0] = 0
# print(z)
print(x)
print(y)

print("ndarray的切片")
x = np.array([1,2,3,4,5])
print(x[2:5])
print(x[:3])
print(x[1:])
print(x[:])
print(x[0:4:2])

x = np.array([[1,2],[3,4],[5,6]])
print(x.shape)
print(x[:2])
print(x[:2,:1])
print(x[:2][:1])

x[:2,:1] = [[0],[6]]
print(x)

print("ndarray的布尔型索引")
x = np.array([3,2,3,1,3,0,4,5,6])
y = np.array([True,True,True,False,False,True,False,False,True])
print(x[y])
print(x[y==False])
print(x>=3)
print(x[x>=3])
print((x==2) | (x==1))
print(x[(x==2) | (x==3)])

print("使用整型数组作为索引")
x = np.array([1,2,3,4,5])
print(x[[0,2,3]])
print(x[[-1,-2,-3]])
x = np.array([[1,2],[3,4],[5,6]])
print(x[[0,1]])
print(x[[0,1],[0,1]])
print("===========")
print(x[[0,1]][:,[0,1]])

print("ndarray的数组转置和轴操作")
a = np.arange(8).reshape((4,2))
print(a)
#转置
print(a.T)
print(np.dot(a,a.T))

#高维数组的轴对象
a = np.arange(24).reshape(2,3,4)
print(a)
#轴变换
# a = a.transpose(2,0,1)
# print(a)
# print(a.shape)
#轴交换
a = a.swapaxes(1,0)
print(a.shape)
#轴交换实现转置
a = np.arange(8).reshape(2,4)
print(a)
print(a.swapaxes(0,1))

print("一元ufunc示例")
x = np.arange(8)
print(x)
print(np.power(x,2))
print(np.sqrt(x))
print(np.square(x))
x = np.array([1.5,1.6,1.7,1.8])
y,z = np.modf(x)
print(y)
print(z)

print("二元ufunc示例")
x = np.array([[1,4],[6,7]])
y = np.array([[2,4],[5,8]])
print(np.maximum(x,y))
print(np.minimum(x,y))
print(x)
print(np.max(x))
print(np.max(x,axis=1))
print(np.mean(x))
print(np.mean(x,axis=0))
print(np.sum(x))
print(np.sum(x,axis=1))

print("where函数")
a = np.array([True,False,False,True])
x = np.where(a,2,3)
print(x)

a = np.array([1,2,3,4])
x = np.where(a>2,-2,2)
print(x)

print(".sort就地排序")
x = np.array([[6,1,3],[1,5,2]])
print(x)
x.sort(axis=0)
print(x)

print("ndarray存取")
x = np.array([[1,6,2],[6,1,3],[1,5,2]])
# np.save("data/file",x)#以二进制文件.npy保存
y = np.load("data/file.npy")
print(y)

print("矩阵求逆")
x = np.array([[1,1],[1,2]])
y = np.linalg.inv(x)
print(y)
print(x.dot(y))
print(np.linalg.det(x))

print("numpy中的随机数")
arr1 = np.random.random((2,3))
arr2 = np.random.randn(2,3)
arr3 = np.random.rand(2,3)
a = np.random.randint(1,10,size=(2,5))
print(arr1)
print(arr2)
print(arr3)
print(a)
b = np.random.normal(6,6,size=(2,5))
print(b)

#抛硬币
x = np.random.randint(0,2,size=10000000)
#正面次数
print((x>0).sum())