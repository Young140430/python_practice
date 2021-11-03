#生成验证码
import random
from PIL import Image,ImageDraw,ImageFont

class GenerateCode:
    #生成随机的内容（A~Z）
    def getcode(self):
        return chr(random.randint(65,90))

    #生成随机的颜色
    def bgcolor(self):
        return (random.randint(90,160),
                random.randint(90,160),
                random.randint(90,160))
    def fontcolor(self):
        return (random.randint(60,120),
                random.randint(60,120),
                random.randint(60,120))

    def gen_img(self):
        w,h = 240,60
        panel = Image.new(size=(w,h),color=(255,255,255),mode="RGB")
        draw = ImageDraw.Draw(panel)
        font = ImageFont.truetype("../data/arial.ttf",size=30)
        for y in range(h):
            for x in range(w):
                draw.point((x,y),fill=self.bgcolor())
        for i in range(4):
            str = self.getcode()
            print(str)
            draw.text((60*i+20,15),text=str,fill=self.fontcolor(),font=font)

        return panel
if __name__ == '__main__':

    genter = GenerateCode()
    img = genter.gen_img()
    img.show()