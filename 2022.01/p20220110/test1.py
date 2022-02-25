from PIL import Image
import numpy as np
import cv2
import os
img_dir=r"F:\XunLeiDownload\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages"
for i in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir,i))
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    l = max(h, w)
    arr = np.zeros((l, l, c), dtype=np.uint8)
    arr[:h, :w] = img[:h, :w]
    cv2.imwrite(r"F:\XunLeiDownload\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages2\{0}".format(i),arr)