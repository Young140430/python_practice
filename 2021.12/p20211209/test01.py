import cv2

img = cv2.imread("img/000002.jpg")
cv2.rectangle(img,(72,  94 ),(72+221, 94+306 ),(0,0,255),2)
cv2.imshow("src",img)
cv2.waitKey(0)
cv2.destroyAllWindows()