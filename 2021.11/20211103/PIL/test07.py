import numpy as np
from PIL import Image

img = Image.open("../img/pic2.jpg")
img_data = np.array(img)
img_data[:,:,0] = 0
img_data[:,:,2] = 0
print(img_data)
R_img = Image.fromarray(img_data)
R_img.show()