import cv2
import numpy as np

image = cv2.imread("img/50.jpg")

dst = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#检测圆
circles = cv2.HoughCircles(dst,cv2.HOUGH_GRADIENT,1,30,param1=40,param2=20,minRadius=30,maxRadius=80)
print(circles)
if not circles is None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)

cv2.imshow("src",image)
cv2.waitKey(0)
cv2.destroyWindow()