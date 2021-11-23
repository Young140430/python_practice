import cv2
import numpy as np

#HSV提取皮肤
def skin_detection_HSV(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 皮肤色调阈值
    lower_blue = np.array([0, 18, 102])
    upper_blue = np.array([17, 133, 242])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    img = cv2.bitwise_and(frame, frame, mask=mask)
    return img

if __name__ == '__main__':
    # cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    # while True:
    #     ret, frame = cap.read()
    #     img = skin_detection_HSV(frame)
    #     cv2.imshow("show HSV", np.hstack((frame, img)))
    #     if cv2.waitKey(42) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    img = cv2.imread("img/ceshi.jpg")
    dst = skin_detection_HSV(img)
    cv2.imshow("src",img)
    cv2.imshow("dst",dst)
    cv2.waitKey(0)
    cv2.destroyWindow()
