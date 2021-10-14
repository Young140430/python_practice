import cv2
img=cv2.imread("E:/1.jpg")
'''cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

c=(0,0,255)
#cv2.line(img,(10,10),(500,500),c,3)
#cv2.rectangle(img,(50,50),(500,500),color=c)
cv2.circle(img,(50,50),50,c)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
