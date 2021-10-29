import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10,10,0.1)
y1 = x**2
y2 = x**3

plt.plot(x,y1)
plt.plot(x,y2)
plt.show()