import cv2
import numpy as np

img = cv2.imread("img/1.jpg")

cv2.line(img,(100,30),(210,180),color=(0,0,255),thickness=2)
cv2.circle(img,(50,50),30,(255,0,0),2)
cv2.rectangle(img,(100,30),(210,180),(0,255,0),2)
cv2.ellipse(img,(100,100),(100,50),0,0,360,(255,0,0),-1)
#多边形
pts = np.array([[50,50],[150,80],[100,180],[0,120]])
cv2.polylines(img,[pts],True,(0,0,255),2)
cv2.putText(img,"hello",(30,330),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),5,lineType=cv2.LINE_AA)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()