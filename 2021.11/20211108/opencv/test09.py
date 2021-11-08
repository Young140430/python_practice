import cv2

src = cv2.imread("../img/1.jpg")

dst = cv2.Canny(src,30,150)
dst2 = cv2.Laplacian(src,-1)

cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.imshow("dst2",dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()