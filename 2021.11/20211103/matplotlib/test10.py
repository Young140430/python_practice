import os
from PIL import Image
import matplotlib.pyplot as plt

plt.ion()
while True:
    for i in os.listdir("../img"):
        img = Image.open(os.path.join("../img/", i))
        plt.clf()
        plt.axis(False)
        plt.imshow(img)
        plt.pause(1)
plt.ioff()
plt.show()