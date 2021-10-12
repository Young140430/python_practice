import PIL.Image as image
import PIL.ImageDraw as draw
import PIL.ImageFont as imgfont
import random
import numpy as np
font=imgfont.truetype("simkai.ttf",60)
img1=image.open("1.jpg")
'''img4=np.array(img1)
print(img4)
#img1.show()
w,h=img1.size
print(w,h)
img2=img1.resize((300,200))
#img2.show()
img3=img2.rotate((90))
#img3.save("2.jpg")
img=draw.Draw(img1)
img.point((200,100),fill="black")#画点
img.rectangle((20,20,100,100),outline="red")#画矩形
img.line((10,50,170,210),fill='green',width=2)#画线，不常用
img.text((120,120),text="我要学人工智能",fill='blue',font=font)#文本
#img1.save("3.jpg")'''
w,h=240,120

def randchar():
    return chr(random.randint(65,90))
print(randchar())
def bg_color():
    return (random.randint(64,255),random.randint(64,255),random.randint(64,255))
print(bg_color())
def f_color():
    return (random.randint(32,128),random.randint(32,128),random.randint(32,128))
print(f_color())
def img():
    return image.new("RGB",(w,h),(255,255,255))
if __name__=='__main__':
    img=img()
    image=draw.Draw(img)
    for x in range(w):
        for y in range(h):
            image.point((x,y),fill=bg_color())
    for i in range(4):
        image.text((60*i+10,30),text=randchar(),fill=f_color(),font=font)
    img.show()