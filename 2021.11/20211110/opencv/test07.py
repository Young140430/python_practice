import cv2

img = cv2.imread(r"img/1.jpg")
img_down = cv2.pyrDown(img)
img_up = cv2.pyrUp(img_down)

img_new = cv2.subtract(img, img_up)
#为了更容易看清楚，做了个提高对比度的操作
img_new = cv2.convertScaleAbs(img_new, alpha=5, beta=0)

cv2.imshow("src",img)
cv2.imshow("img_up",img_up)
cv2.imshow("img_new",img_new)
cv2.waitKey(0)
cv2.destroyWindow()