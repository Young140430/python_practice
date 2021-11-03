from PIL import Image

img1 = Image.open("../img/cat.jpg")
img2 = Image.open("../img/pic2.jpg")

img2.paste(img1,(100,100))
img2.show()