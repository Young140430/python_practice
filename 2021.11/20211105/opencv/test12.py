import cv2

img = cv2.imread("../img/1.jpg")

h,w,c = img.shape

dst1 = cv2.resize(img,(w//2,h//2))
dst2 = cv2.transpose(img)
dst3 = cv2.flip(img,1)

cv2.imshow("src",img)
cv2.imshow("dst1",dst1)
cv2.imshow("dst2",dst2)
cv2.imshow("dst3",dst3)
cv2.waitKey(0)
cv2.destroyAllWindows()