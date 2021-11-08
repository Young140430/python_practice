import cv2
import numpy as np

src = cv2.imread("../img/17.jpg")
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 边界矩形
x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 最小矩形
rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
img_contour = cv2.drawContours(src, [box], 0, (0, 255, 0), 2)

# 最小外切圆
(x, y), radius = cv2.minEnclosingCircle(contours[0])
center = (int(x), int(y))
radius = int(radius)
img_contour = cv2.circle(src, center, radius, (255, 0, 0), 2)

#边界矩形的宽高比
x,y,w,h = cv2.boundingRect(contours[0])
aspect_ratio = float(w)/h
print(aspect_ratio)

#轮廓面积与边界矩形面积的比
area = cv2.contourArea(contours[0])
x,y,w,h = cv2.boundingRect(contours[0])
rect_area = w*h
extent = float(area)/rect_area
print(extent)

#轮廓面积与凸包面积的比
area = cv2.contourArea(contours[0])
hull = cv2.convexHull(contours[0])
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area
print(solidity)

#与轮廓面积相等的圆形的直径
area = cv2.contourArea(contours[0])
equi_diameter = np.sqrt(4*area/np.pi)
print(equi_diameter)

#对象的方向
(x,y),(MA,ma),angle = cv2.fitEllipse(contours[0])
print(angle)


cv2.drawContours(src,contours,0,(0,0,255),3)
cv2.imshow("src",src)
cv2.waitKey(0)
cv2.destroyAllWindows()