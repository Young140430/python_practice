import cv2
import numpy as np

src = cv2.imread("img/16.jpg")


std = cv2.Canny(src,100,350)

lines_p = cv2.HoughLinesP(std,1,np.pi/180,threshold=80)

#绘制直线
for i in range(len(lines_p)):
    x1,y1,x2,y2 = lines_p[i][0]
    cv2.line(src,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow("std",std)
cv2.imshow("src",src)
cv2.waitKey(0)
cv2.destroyWindow()