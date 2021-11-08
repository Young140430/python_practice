import cv2
import numpy as np

src = cv2.imread("../img/1.jpg")

kernel = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]],np.float32)

dst =cv2.filter2D(src,-1,kernel)

cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()