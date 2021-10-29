import os
import PIL.Image as img
import numpy as np

image_path=r"E:/imgs"

class Sample:
    def read_data(self):
        self.img_arr=[]
        for name in os.listdir(image_path):
            imgs=img.open(r"{0}/{1}".format(image_path,name))
            images=(np.array(imgs)/255-0.5)*2
            self.img_arr.append(images)
        return self.img_arr

    def get_batch(self,set):
        self.read_data()
        self.get_arr=[]
        for i in range(set):
            num=np.random.randint(len(self.img_arr))
            print(len(self.img_arr))
            imge=self.img_arr[num]
            #print(imge)
            self.get_arr.append(imge)
        return self.get_arr

sample=Sample()
#sample.read_data()
print(sample.get_batch(1))