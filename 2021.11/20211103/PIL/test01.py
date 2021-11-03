import numpy as np
from PIL import Image

img_data = np.array([[[255,0,255]]],dtype=np.uint8)
print(img_data.shape)
img = Image.fromarray(img_data)
img.show()