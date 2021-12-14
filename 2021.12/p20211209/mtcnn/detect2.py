import torch
from PIL import Image
from PIL import ImageDraw,ImageFont
import numpy as np
import utils2
import nets
from torchvision import transforms
import time
import os
import math


# 网络调参
# P网络:
p_cls = 0.6 #原为0.6
p_nms = 0.5 #原为0.5
# R网络：
r_cls = 0.6 #原为0.6
r_nms = 0.5 #原为0.5
# O网络：
o_cls = 0.97 #原为0.97
o_nms = 0.7 #原为0.7

class Detector():
    def __init__(self, pnet_param="param/pnet.pt", rnet_param="param/rnet.pt", onet_param="param/onet.pt",
                 isCuda=True):
        self.isCuda=isCuda
        self.pnet=nets.PNet()
        self.rnet=nets.RNet()
        self.onet=nets.ONet()
        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.__image_transform=transforms.Compose([transforms.ToTensor()])
    def detect(self,image):
        pass
    def __pnet_detect(self,image):
        boxes=[]
        img=image
        w,h=img.size
        min_side_len=min(w,h)
        scale=1
        while min_side_len>12:
            img_data=self.__image_transform(img)
            if self.isCuda:
                img_data=img_data.cuda()
            img_data.unsqueeze_(0)
            _cls,_offset=self.pnet(img_data)
            cls=_cls[0][0].cpu().data
            offset=_offset[0].cpu().data
            idxs=torch.nonzero(torch.gt(cls,p_cls))
            boxes.append(self.__box(idxs,offset,cls[idxs[:,1],idxs[:,0]],scale))
            scale*=0.7
            _w = int(w * scale)
            _h = int(h * scale)
            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)
            return utils2.nms(np.array(boxes),p_nms)
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        _x1 = (start_index[:,1].float() * stride) / scale  # 索引乘以步长，除以缩放比例；★特征反算时“行索引，索引互换”，原为[0]
        _y1 = (start_index[:,0].float() * stride) / scale
        _x2 = (start_index[:,1].float() * stride + side_len - 1) / scale
        _y2 = (start_index[:,0].float() * stride + side_len - 1) / scale

        ow = _x2 - _x1  # 人脸所在区域建议框的宽和高
        oh = _y2 - _y1

        _offset = offset[:, start_index[:,0], start_index[:,1]]  # 根据idxs行索引与列索引，找到对应偏移量△δ:[x1,y1,x2,y2]
        x1 = _x1 + ow * _offset[0]  # 根据偏移量算实际框的位置，x1=x1_+w*△δ；生样时为:△δ=x1-x1_/w
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]  # 正式框：返回4个坐标点和1个偏移量
    def __rnet_detect(self, image, pnet_boxes):
        pass
    def __onet_detect(self, image, rnet_boxes):
        pass

if __name__ == '__main__':
    font = ImageFont.truetype("font/arial.ttf", size=23)
    image_path = r"test_images"