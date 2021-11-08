import cv2

src = cv2.imread("../img/17.jpg")
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

epsilon = 13 #误差
approx = cv2.approxPolyDP(contours[0],epsilon,True)
print(approx)
cv2.drawContours(src,[approx],-1,(0,0,255),3)
cv2.imshow("src",src)
cv2.waitKey(0)
cv2.destroyAllWindows()