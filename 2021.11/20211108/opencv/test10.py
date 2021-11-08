import cv2

src = cv2.imread("../img/24.jpg")
cv2.imshow("src",src)
#增亮
src = cv2.convertScaleAbs(src,alpha=6,beta=5)
dst = cv2.GaussianBlur(src,(5,5),1)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
dst = cv2.morphologyEx(dst,cv2.MORPH_OPEN,kernel)
dst = cv2.morphologyEx(dst,cv2.MORPH_CLOSE,kernel)
cv2.imshow("s1",dst)
dst = cv2.Canny(dst,80,150)

cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()