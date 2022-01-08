from torchvision import models
from torch import nn
import torch
from torch.nn.functional import normalize


class arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn(feature_num, cls_num))

    def forward(self, features, s=1, m=1):
        x = normalize(features, dim=1)
        w = normalize(self.w, dim=0)
        cosa = torch.matmul(x, w) / 10
        a = torch.acos(cosa)

        out = torch.exp(s * torch.cos(a + m) * 10) / (
                torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(s * cosa * 10) + torch.exp(
            s * torch.cos(a + m) * 10))

        return torch.log(out)
class facenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_net=nn.Sequential(
            models.densenet121(pretrained=True)
        )
        self.feat_net=nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(1000,512,bias=False)
        )
        self.arcface=arcsoftmax(512,11)

    def forward(self,xs):
        sub=self.sub_net(xs)
        feat=self.feat_net(sub)
        return self.arcface(feat)

    def encode(self,xs):
        return self.feat_net(self.sub_net(xs))












