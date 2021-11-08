import cv2

src = cv2.imread("../img/5.jpg")

dst = cv2.GaussianBlur(src,(7,7),6)
#专门用来去掉椒盐噪声
dst1 = cv2.medianBlur(src,3)

cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.imshow("dst1",dst1)
cv2.waitKey(0)
cv2.destroyAllWindows()