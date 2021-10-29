import numpy as np
import random
a=np.floor(10*random.random())
print(a)
b=np.array([5,9,7,3,1])
c=np.array([8,2,6,4,10])
d=np.column_stack((b,c))
e=np.hstack((b,c))
print(d)
print(e)