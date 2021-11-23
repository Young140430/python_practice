import cv2
import numpy as np

A = cv2.imread("img/21.jpg")
B = cv2.imread("img/22.jpg")

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    # cv2.imshow(f"win{i}",G)
    gpA.append(G)
# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    # cv2.imshow(f"win{i}",G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i - 1], GE)
    # L = cv2.convertScaleAbs(L, alpha=8, beta=0)
    # cv2.imshow(f"win{i}",L)
    lpA.append(L)

lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i - 1], GE)
    # L = cv2.convertScaleAbs(L, alpha=8, beta=0)
    # cv2.imshow(f"win{i}",L)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
    # ls = cv2.convertScaleAbs(ls, alpha=8, beta=0)
    # cv2.imshow("win",ls)
    LS.append(ls)

ls_ = LS[0]
for i in range(1, 6):
    ls_1 = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_1, LS[i])
    cv2.imshow(f"win{i}",ls_)
    cv2.imshow(f"wing{i}",ls_1)

cv2.waitKey(0)
