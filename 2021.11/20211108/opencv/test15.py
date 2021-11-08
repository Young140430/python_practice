import cv2

src = cv2.imread("../img/23.jpg")

#高斯模糊去噪
src = cv2.GaussianBlur(src,(3,3),1)
#灰度化
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
#梯度图
sobel_x = cv2.Sobel(gray,cv2.CV_16S,1,0)
#高亮，并转回8位
absx = cv2.convertScaleAbs(sobel_x)
#二值化
ret,binary = cv2.threshold(absx,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#闭操作：闭操作可以将目标区域连接成一个整体
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT,(17,5))
image = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernelX)

#去噪
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT,(20,1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT,(1,20))
image = cv2.dilate(image,kernelX)
image = cv2.erode(image,kernelX)
image = cv2.erode(image,kernelY)
image = cv2.dilate(image,kernelY)

#平滑处理，中值滤波
image = cv2.medianBlur(image,15)
#查找轮廓
contours,_ = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for item in contours:
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    if w>2*h:
        #裁剪区域图片
        chepai = src[y:y+h,x:x+w]
        cv2.imshow("chepai"+str(x),chepai)
#绘制轮廓
cv2.drawContours(src,contours,-1,(0,0,255),2)
cv2.imshow("src",src)
cv2.waitKey(0)
cv2.destroyAllWindows()

#1、实现红绿灯识别
#2、皮肤检测（检测视屏中的的人体皮肤）
#3、车道检测