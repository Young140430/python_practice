import numpy as np
import matplotlib.pyplot as plt
x=np.arange(-10,10,0.1)
x1=np.arange(0.1,10,0.1)
y1=x
y2=x**2
y3=x**3
y4=np.log(x1)/np.log(2)
y5=x1**(-1)
'''plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)'''
plt.plot(x1,y5)
plt.plot(x1,y4)
plt.show()