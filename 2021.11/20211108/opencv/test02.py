import cv2
import numpy as np

src = cv2.imread("../img/4.jpg")

dst1 = cv2.blur(src,(5,5))
dst2 = cv2.GaussianBlur(src,(5,5),0)


cv2.imshow("src",src)
cv2.imshow("dst1",dst1)
cv2.imshow("dst2",dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()