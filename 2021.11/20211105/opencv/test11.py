import cv2

img1 = cv2.imread("img/1.jpg")
img2 = cv2.imread("img/6.jpg")
print(img2.shape)

dst_and = cv2.bitwise_and(img1,img2)
dst_or = cv2.bitwise_or(img1,img2)
dst_xor = cv2.bitwise_xor(img1,img2)
dst_not = cv2.bitwise_not(img1)

cv2.imshow("src1",img1)
cv2.imshow("src2",img2)
cv2.imshow("dst_and",dst_and)
cv2.imshow("dst_or",dst_or)
cv2.imshow("dst_xor",dst_xor)
cv2.imshow("dst_not",dst_not)
cv2.waitKey(0)
cv2.destroyAllWindows()