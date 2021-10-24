import random

_x=[1,3,5,7,9]
_y=[6,16,26,36,46]

w=random.random()
b=random.random()

for i in range(100):
    for x,y in zip(_x,_y):
        z=w*x+b
        o=z-y
        loss=o**2
        dw=-2*o*x
        db=-2*o
        w=w+0.01*dw
        b=b+0.01*db
        print(w,b)