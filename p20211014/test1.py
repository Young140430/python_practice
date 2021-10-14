import matplotlib.pyplot as plt
import numpy as np
'''x=np.linspace(0,2*np.pi,100)
y1=np.sin(x)
y2=np.cos(x)
plt.title("sinx&cosx")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y1)
plt.plot(x,y2)
plt.show()'''

'''name_list=['A','B','C','D']
num_list=[1,3,5,4]
num_list2=[9,5,7,3]
x=list(range(len(name_list)))
width=0.4
plt.bar(x,num_list,width=width,color="red",tick_label=name_list)
for i in range(4):
    x[i]=x[i]+width
plt.bar(x,num_list2,width=width,color="green",tick_label=name_list)
plt.show()'''

'''label=['A','B','C','D']
num=[28,75,57,12]
ex=[0,0.1,0,0]
plt.pie(x=num,autopct='%.2f%%',explode=ex,labels=label,colors="rgby",shadow=True,startangle=30)
plt.show()'''

ax=[]
ay=[]
plt.ion()
for i in range(100):
    ax.append(i)
    ay.append(i**3)
    plt.clf()
    plt.plot(ax,ay)
    plt.pause(0.1)
plt.ioff()