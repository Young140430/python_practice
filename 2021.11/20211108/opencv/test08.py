import cv2
import numpy as np

src = cv2.imread("../img/2.jpg")

dst1 = cv2.Laplacian(src,-1)

kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.float32)
dst2 = cv2.filter2D(src,-1,kernel)

cv2.imshow("src",src)
cv2.imshow("dts1",dst1)
cv2.imshow("dst2",dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()