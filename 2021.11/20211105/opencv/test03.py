import cv2

cap = cv2.VideoCapture("../img/android.mp4")
#cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("")

while True:
    ret,frame = cap.read()
    if ret:
        cv2.imshow("video",frame)
        if cv2.waitKey(1000//24) & 0xFF== ord("q"):
            break
    else:
        #print("未读取到视频信息！")
        break
cap.release()
cv2.destroyAllWindows()