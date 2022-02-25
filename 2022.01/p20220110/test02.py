import thop
import torch
from thop import clever_format
from torch import nn
from torch.nn import functional
import time
class UpsampleLayer(nn.Module):
    def __init__(self):
        super(UpsampleLayer, self).__init__()
    def forward(self,x):
        return functional.interpolate(x,scale_factor=2,mode="nearest")

class ConvolutionalLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_mode = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    def forward(self,x):
        return self.sub_mode(x)

class ResidualLayer(nn.Module):
    def __init__(self,in_channels):
        super(ResidualLayer, self).__init__()
        self.sub_mode = nn.Sequential(
            ConvolutionalLayer(in_channels,in_channels//2,1,1,0),
            ConvolutionalLayer(in_channels//2,in_channels,3,1,1)
        )
    def forward(self,x):
        return self.sub_mode(x) + x

class DownsamplingLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownsamplingLayer, self).__init__()

        self.sub_mode = nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,3,2,1)
        )
    def forward(self,x):
        return self.sub_mode(x)

class ConvolutionalSet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvolutionalSet, self).__init__()

        self.sub_mode = nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,1,1,0),
            ConvolutionalLayer(out_channels,in_channels,3,1,1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)
        )
    def forward(self,x):
        return self.sub_mode(x)

class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()

        self.trunk_52 = nn.Sequential(
            ConvolutionalLayer(3,32,3,1,1),
            DownsamplingLayer(32,64),

            ResidualLayer(64),
            DownsamplingLayer(64,128),

            ResidualLayer(128),
            ResidualLayer(128),
            DownsamplingLayer(128, 256),

            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256)
        )

        self.trunk_26 = nn.Sequential(
            DownsamplingLayer(256,512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512)
        )

        self.trunk_13 = nn.Sequential(
            DownsamplingLayer(512,1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )

        self.convset_13 = nn.Sequential(
            ConvolutionalSet(1024,512)
        )

        self.detetion_13 = nn.Sequential(
            ConvolutionalLayer(512,1024,3,1,1),
            nn.Conv2d(1024,3*25,1,1,0)
        )
        self.up_26 = nn.Sequential(
            ConvolutionalLayer(512,256,1,1,0),
            UpsampleLayer()
        )

        self.convset_26 = nn.Sequential(
            ConvolutionalSet(768,256)
        )
        self.detetion_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, 3*25, 1, 1, 0)
        )

        self.up_52 = nn.Sequential(
            ConvolutionalLayer(256,128,1,1,0),
            UpsampleLayer()
        )
        self.convset_52 = nn.Sequential(
            ConvolutionalSet(384,128)
        )
        self.detetion_52 = nn.Sequential(
            ConvolutionalLayer(128,256,3,1,1),
            nn.Conv2d(256,3*25,1,1,0)
        )
    def forward(self,x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_out_13 = self.convset_13(h_13)
        detetion_13 = self.detetion_13(convset_out_13)

        up_out_26 = self.up_26(convset_out_13)
        route_out_26 = torch.cat((up_out_26,h_26),dim=1)
        convset_out_26 = self.convset_26(route_out_26)
        detetion_26 = self.detetion_26(convset_out_26)

        up_out_52 = self.up_52(convset_out_26)
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)
        convset_out_52 = self.convset_52(route_out_52)
        detetion_52 = self.detetion_52(convset_out_52)

        return detetion_13,detetion_26,detetion_52
if __name__ == '__main__':
    yolo = MainNet()
    x = torch.randn(1,3,416,416)
    start_time = time.time()
    y_13,y_26,y_52 = yolo(x)
    end_time = time.time()
    print(end_time-start_time)#1.0398929119110107
    print(clever_format(thop.profile(yolo, (x,))))#('34.60G', '63.27M')
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)
