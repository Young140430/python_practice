import cv2

src = cv2.imread("../img/1.jpg")

dst = cv2.GaussianBlur(src,(5,5),0)
dst = cv2.addWeighted(src,2,dst,-1,0)
dst1 = cv2.Laplacian(src,-1)


cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.imshow("dst1",dst1)
cv2.waitKey(0)
cv2.destroyAllWindows()