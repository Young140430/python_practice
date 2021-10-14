import numpy as np
'''a=np.array([[1,2],[3,4],[5,6]])
b=a.reshape(2,3)
c=a.dot(b)
d=b.dot(a)
print(c)
print(d)'''

#a=np.ndarray([12]).reshape(2,2,3)
#print(a)

#a=np.zeros(shape=[4,10],dtype=np.float32)
#print(a)
#b=np.ones(shape=[4,10],dtype=np.float32)
#print(b)
#c=np.empty(shape=[4,10],dtype=np.float32)
#print(c)

'''a=np.zeros(shape=[4,10],dtype=np.float32)
for i,k in enumerate(a):
    if i==0:
        k[8]=1
    elif i==1:
        k[5]=1
    elif i==2:
        k[2]=1
    else:
        k[6]=1
print(a)
c=np.argmax(a,axis=1)
print(c)'''

a=np.arange(12).reshape(4,3)
print(a)
print(a.T)