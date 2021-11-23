import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img/45.jpg")

img_B = cv2.calcHist(img,[0],None,[256],[0,256])
plt.plot(img_B,label="B",color="b")

img_G = cv2.calcHist(img,[1],None,[256],[0,256])
plt.plot(img_G,label="G",color="g")

img_R = cv2.calcHist(img,[2],None,[256],[0,256])
plt.plot(img_R,label="R",color="r")

cv2.imshow("src",img)

plt.legend()
plt.show()

cv2.waitKey(0)
