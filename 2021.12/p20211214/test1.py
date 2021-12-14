import cv2

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if ret:
        cv2.imshow("video",frame)
        if cv2.waitKey(1000//24) & 0xff==ord("q"):
            break
    else:
        print("未读取到视频信息")
        break
cap.release()
cv2.destroyAllWindows()
