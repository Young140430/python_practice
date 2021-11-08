import cv2

img = cv2.imread("../img/3.jpg",0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# dst = cv2.dilate(img,kernel)
dst = cv2.erode(img,kernel)

cv2.imshow("src",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()