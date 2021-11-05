import cv2
import numpy as np

# img = np.random.randint(0,255,size=200*300*3,dtype=np.uint8).reshape(200,300,3)
# cv2.imshow("img",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite("img/img_save.jpg",img)
img = np.zeros((200,300,3),dtype=np.uint8)
img[...,0]=0
img[...,1]=0
img[...,2]=255
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
