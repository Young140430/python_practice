import cv2

src = cv2.imread("../img/17.jpg")
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#检测并修复凸性缺陷
hull = cv2.convexHull(contours[0])
#isContourConvex仅仅判断是否是凸的
print(cv2.isContourConvex(contours[0]), cv2.isContourConvex(hull))

cv2.drawContours(src,[hull],0,(0,0,255),3)
cv2.imshow("src",src)
cv2.waitKey(0)
cv2.destroyAllWindows()