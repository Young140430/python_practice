import cv2

src = cv2.imread("../img/2.jpg")

S_x = cv2.Sobel(src,-1,1,0)
S_y = cv2.Sobel(src,-1,0,1)
Sc_x = cv2.Scharr(src,-1,1,0)
Sc_y = cv2.Scharr(src,-1,0,1)
cv2.imshow("src",src)
cv2.imshow("S_x",S_x)
cv2.imshow("S_y",S_y)
cv2.imshow("Sc_x",Sc_x)
cv2.imshow("Sc_y",Sc_y)

cv2.waitKey(0)
cv2.destroyAllWindows()