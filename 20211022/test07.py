import numpy as np


a = np.array([[1,2,3],[4,5,6]])
print(a.shape)

b = a.T
print(b.shape)

c = a.dot(b)
print(c.shape)