import cv2

src = cv2.imread("../img/26.jpg")

gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contours[0].shape)
print(contours)
M = cv2.moments(contours[0]) # 矩
cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
print("重心:", cx, cy)

area = cv2.contourArea(contours[0])
print("面积:", area)

perimeter = cv2.arcLength(contours[0], True)
print("周长:", perimeter)


cv2.drawContours(src,contours,-1,(0,0,255),3)

cv2.imshow("src",src)
cv2.waitKey(0)
cv2.destroyAllWindows()