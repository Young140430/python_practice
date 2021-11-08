import cv2

img1 = cv2.imread("../img/1.jpg")
img2 = cv2.imread("../img/6.jpg")

dst1 = cv2.add(img1,img2)
dst2 = cv2.addWeighted(img1,0.7,img2,0.3,0)

cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("dst",dst1)
cv2.imshow("dst2",dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()