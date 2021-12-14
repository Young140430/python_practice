import os

import PIL
from PIL import Image
import numpy as np

bg_path = "bg_pic/0"
x = 1
for filename in os.listdir(bg_path):
    print(filename)
    try:
        background = Image.open("{0}/{1}".format(bg_path, filename))
        shape = np.shape(background)
        if len(shape) == 3 and shape[0] > 100 and shape[1] > 100:
            background = background
        else:
            continue
        background_resize = background.resize((300, 300))
        background_resize = background_resize.convert("RGB")


        background_resize.save("bg_pic/0-1/{0}{1}.png".format(x, "." + str(0) + "." + str(0) +"." + str(0) + "." + str(0)))
        x += 1
    except PIL.UnidentifiedImageError:
        continue