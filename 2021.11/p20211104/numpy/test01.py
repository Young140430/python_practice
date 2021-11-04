import numpy as np
a=np.arange(1,49).reshape(2,2,3,2,2)
print(a)
print("=============================")
print(a.transpose(0,2,1,3,4))
print("=============================")
print(a.reshape(4,3,2,2))