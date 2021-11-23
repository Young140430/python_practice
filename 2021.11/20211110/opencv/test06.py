import cv2

img = cv2.imread("img/1.jpg")

for i in range(3):
    cv2.imshow(f"img{i}",img)
    img = cv2.pyrUp(img)
cv2.waitKey(0)
cv2.destroyWindow()