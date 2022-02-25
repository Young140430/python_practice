from __future__ import division
import time
import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import clever_format
from torch.autograd import Variable
import numpy as np
from InvertedResidual import InvertedResidual, extend_layers, output_layers, conv_dbl, con1x1
from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv0 = conv_dbl(3, 32, 2)  # First DownSample 416 -> 208
        self.trunk52 = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),  # Second DownSample 208 -> 104
            InvertedResidual(24, 24, 1, 6),
            InvertedResidual(24, 32, 2, 6),  # Third DownSample 104 -> 52
            InvertedResidual(32, 32, 1, 6),
        )
        self.trunk26 = nn.Sequential(
            InvertedResidual(32, 64, 2, 6),  # Fourth DownSample 52 -> 26
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
        )
        self.trunk13 = nn.Sequential(
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 160, 2, 6),  # Fifth DownSample 26 -> 13
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6),
        )

        self.conEt1 = extend_layers(160, 512)
        self.conOp1 = output_layers(512, 75)
        self.conUp1 = nn.Sequential(con1x1(512, 256), nn.ConvTranspose2d(256, 256, 3, 2, 1, 1, 256))

        self.conEt2 = extend_layers(320, 256)
        self.conOp2 = output_layers(256, 75)
        self.conUp2 = nn.Sequential(con1x1(256, 128), nn.ConvTranspose2d(128, 128, 3, 2, 1, 1, 128))

        self.conEt3 = extend_layers(160, 256)
        self.conOp3 = output_layers(256, 75)

        '''self.yolo13 = YOLOLayer([(116, 90), (156, 198), (373, 326)], 80, 416)
        self.yolo26 = YOLOLayer([(30, 61), (62, 45), (59, 119)], 80, 416)
        self.yolo52 = YOLOLayer([(10, 13), (16, 30), (33, 23)], 80, 416)'''

    def forward(self, x, target=None):
        img_dim = x.shape[2]
        x = self.conv0(x)
        x = self.trunk52(x)
        xR52 = x
        x = self.trunk26(x)
        xR26 = x
        x = self.trunk13(x)
        x = self.conEt1(x)
        xOp13 = self.conOp1(x)
        x = self.conUp1(x)
        x = torch.cat([x, xR26], 1)

        x = self.conEt2(x)
        xOp26 = self.conOp2(x)
        x = self.conUp2(x)
        x = torch.cat([x, xR52], 1)
        x = self.conEt3(x)
        xOp52 = self.conOp3(x)

        '''out13, loss13 = self.yolo13(xOp13, target, img_dim)
        out26, loss26 = self.yolo26(xOp26, target, img_dim)
        out52, loss52 = self.yolo52(xOp52, target, img_dim)'''

        return xOp13,xOp26,xOp52

if __name__ == '__main__':
    input = torch.randn(1, 3, 416, 416)
    model = MobileNet()
    start_time = time.time()
    output13,output26,output52 = model(input)
    end_time = time.time()
    print(end_time - start_time)#0.39995503425598145
    # cs, params = thop.profile(model, (input,))
    # print(cs)
    # print(params)
    print(clever_format(thop.profile(model, (input,))))#('3.97G', '5.84M')
    print(output13.shape)
    print(output26.shape)
    print(output52.shape)