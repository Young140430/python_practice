import numpy as np

x = np.array([[3],[1],[6]])

y = 4 * x

print(x)
print(y)

print("==============")
print(np.linalg.inv(x.T@x)@x.T@y)