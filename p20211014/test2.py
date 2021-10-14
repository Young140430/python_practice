import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
'''x=np.random.normal(0,1,100)
y=np.random.normal(0,1,100)
z=np.random.normal(0,1,100)
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x,y,z)
plt.show()'''

n=1000
x=np.random.randn(n)
y=np.random.randn(n)
plt.scatter(x,y)
plt.show()