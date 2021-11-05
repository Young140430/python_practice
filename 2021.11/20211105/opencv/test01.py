import cv2


img = cv2.imread("../img/1.jpg")
print(type(img))
print(img.shape)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()