import random
import matplotlib.pyplot as plt

_x = [i/100 for i in range(100)]
_y = [3*e+4+random.random() for e in _x]

# print(_x)
# print(_y)
# plt.plot(_x,_y,".")
# plt.show()
w= random.random()
b = random.random()

for i in range(30):
    for x, y in zip(_x, _y):
        z = x * w + b
        o = z - y
        loss = o ** 2

        dw = -2 * o * x
        db = -2 * o

        w = w + 0.1 * dw
        b = b + 0.1 * db

        print(w)
        print(b)
plt.plot(_x,_y,".")
v = [w*e+b for e in _x]
plt.plot(_x,v)
plt.show()