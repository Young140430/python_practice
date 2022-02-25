import cv2
import torchvision

from module import *
import cfg
from utils import nms
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        self.net = Darknet53()
        self.net.load_state_dict(torch.load(r"E:\yolov3_param\voc_best3.pt"))
        self.net.eval()

    def forward(self, input, thresh, anchors):
        input=transforms(input)
        input=input.unsqueeze(dim=0)
        output_13, output_26, output_52 = self.net(input)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        #output:N,H,W,3,15
        #mask:N,H,W,3
        mask = output[..., 0] > thresh

        idxs = mask.nonzero()
        vecs = output[mask]
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        anchors = torch.Tensor(anchors)

        n = idxs[:, 0]  # 所属的图片
        a = idxs[:, 3]  # 建议框

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # 原图的中心点x

        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])

        return torch.stack([n.float(), cx, cy, w, h], dim=1)


if __name__ == '__main__':
    detector = Detector()
    for i in range(1, 16):
        data = cv2.imread(f"yolov3_test/{i}.jpg")
        data1 = cv2.imread(f"yolov3_test2/{i}.jpg")
        h, w, c = data1.shape[0],data1.shape[1],data1.shape[2]
        l = max(h,w)
        data2 = cv2.resize(data, (416, 416))
        y = detector(data2, 0, cfg.ANCHORS_GROUP)
        for line in y:
            cls, _x, _y, _w, _h = line[0].item(), line[1].item(), line[2].item(), line[3].item(), line[4].item()
            _x, _y, _w, _h = _x / 416 * l, _y / 416 * l, _w / 416 * l, _h / 416 * l
            x1, y1, x2, y2 = _x - _w / 2, _y - _h / 2, _x + _w / 2, _y + _h / 2
            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
            cv2.rectangle(data, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imshow("img", data)
        print(y)
        print(y.shape)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


