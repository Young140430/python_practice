import numpy as np
a=np.arange(1,49).reshape(2,2,3,2,2)
print(a)
print(a[:,:,[0,2],[0,1],[1,0]])