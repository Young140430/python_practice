import numpy as np
a=np.arange(5,25).reshape(5,4)
print(a)
b=np.arange(5,25).reshape(4,5)
print(b)
c=a@b
d=b@a
print(c)
print(d)
a=np.arange(1,121).reshape(1,2,3,4,5)
print(a)
