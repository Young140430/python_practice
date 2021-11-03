from PIL import Image,ImageFilter

img = Image.open("../img/pic3.jpg")
# print(img)
# pixel = img.getpixel((300,400))
# print(pixel)
# img.show()
# w,h = img.size
# print(w,h)
# mode = img.mode
# print(mode)
# #转换图像的色彩模式
# img = img.convert("L")
# pixel = img.getpixel((300,400))
# print(pixel)
# img.show()
# #转换图像的色彩模式
# img = img.convert("RGB")
# print(img.mode)
# pixel = img.getpixel((300,400))
# print(pixel)
# img.show()
# #缩放图像
# w,h = img.size
# #重新采样 resample=Image.ANTIALIAS  消除锯齿
# img = img.resize((w//2,h//2),resample=Image.ANTIALIAS)
# print(img.size)
# #旋转
# img = img.rotate(-45)
# img.show()
#抠图
# img = img.crop((150,300,260,400))
# img.show()

#滤波器
img.show()
img = img.filter(ImageFilter.CONTOUR)
# img = img.filter(ImageFilter.BLUR)
#img = img.filter(ImageFilter.DETAIL)
#img = img.filter(ImageFilter.EMBOSS)
img.show()
#保存图像
img.save("../img/save.jpg")