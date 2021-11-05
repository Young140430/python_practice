import cv2
from PIL import Image

img = cv2.imread("img/1.jpg")
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img = Image.fromarray(img)
# img.show()
dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("src",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()