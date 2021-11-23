import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img/18.jpg",0)
his = cv2.calcHist(img,[0],None,[256],[0,256])
plt.plot(his,color="r")

cv2.imshow("src",img)

dst1 = cv2.equalizeHist(img)
cv2.imshow("dst1",dst1)
his = cv2.calcHist(dst1,[0],None,[256],[0,256])
plt.plot(his,color="g")
#直方图自适应均衡化
op = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
dst2 = op.apply(img)
cv2.imshow("dst2",dst2)
his = cv2.calcHist(dst2,[0],None,[256],[0,256])
plt.plot(his,color="b")
plt.show()

cv2.waitKey(0)
cv2.destroyWindow()