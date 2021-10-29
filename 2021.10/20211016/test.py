import PIL.Image as image
import PIL.ImageDraw as draw
img1=image.open("1.jpg")
img=draw.Draw(img1)
img.rectangle((149.53,84.11,193.98,128.56),outline="green")
img.rectangle((117.33,100.77,161.78,145.21),outline="green")
img.rectangle((153.97,181.82,198.42,224.00),outline="green")
img.rectangle((211.71,156.41,224.00,196.16),outline="green")
img1.save("2.jpg")