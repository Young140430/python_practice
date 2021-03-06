import cv2

img = cv2.imread("../img/4.jpg",0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# dst = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
# dst = cv2.erode(img,kernel)
# dst = cv2.dilate(dst,kernel)
# dst = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
# dst = cv2.dilate(img,kernel)
# dst = cv2.erode(dst,kernel)
# dst = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
# dst = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
dst = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)


cv2.imshow("src",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()