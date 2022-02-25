import cv2
import cfg,cfg2,cfg3
import math
from model import *
from module import *
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import os
import glob
from utils import nms

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def Net_out(image_data_):
    # net_ = Yolo_Net()
    net_ = MobileNet()
    # print(net_)
    net_.load_state_dict(torch.load(r"E:\yolov3_param\pool_car_best.pt"))
    net_.eval()
    return net_(image_data_)


def get_offsets(output, thresh):
    output = output.permute(0, 2, 3, 1)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    output[..., 0] = torch.sigmoid(output[..., 0])
    mask = output[..., 0] > thresh
    indexs = mask.nonzero()
    offsets = output[mask]
    return indexs, offsets


def get_boxes(index, offset, t_size, resize_t, anchors):
    anchors = torch.Tensor(anchors)
    a = index[:, 3]
    cy = (index[:, 1].float() + offset[:, 2]) * t_size / resize_t
    cx = (index[:, 2].float() + offset[:, 1]) * t_size / resize_t
    w = anchors[a, 0] * torch.exp(offset[:, 3]) / resize_t
    h = anchors[a, 1] * torch.exp(offset[:, 4]) / resize_t
    class_index = torch.argmax(offset[:, 5:], dim=1)
    print(class_index)
    # print(offset[:, 0])
    return torch.stack([cx, cy, w, h, offset[:, 0], class_index], dim=1)


def box_change(boxes):
    if boxes.shape[0]==0:
        return torch.Tensor([])
    boxes[:, 0] = boxes[:, 0].int() - torch.div(boxes[:, 2].int(), 2, rounding_mode="trunc")
    boxes[:, 1] = boxes[:, 1].int() - torch.div(boxes[:, 3].int(), 2, rounding_mode="trunc")
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2].int()
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3].int()
    return boxes


if __name__ == '__main__':

    cls_names = ['car ','pool ']

    path = r"E:\swimming-pool-and-car\test_data_images\test_data_images\images"
    for img_file in glob.glob("{}/*.jpg".format(path)):
        print(img_file)
        image = cv2.imread(img_file)
        img_h, img_w = image.shape[0], image.shape[1]
        if img_h != img_w:
            side_len = max(img_h, img_w)
        else:
            side_len = img_h
        image_new = np.zeros((side_len, side_len, 3), dtype=np.uint8)
        image_new[:img_h, :img_w] = image
        img_h_new, img_w_new = image_new.shape[0], image_new.shape[1]
        image_new = cv2.cvtColor(image_new, cv2.COLOR_BGR2RGB)
        image_new = cv2.resize(image_new, (416, 416))
        resize_w = 416 / img_w_new
        resize_h = 416 / img_h_new
        image_data = transforms(image_new)
        image_data = torch.unsqueeze(image_data, dim=0)
        # print(image_data.shape)
        output_13, output_26, output_52 = Net_out(image_data)

        idxs_13, vecs_13 = get_offsets(output_13, 0.1)
        vecs_13[..., 5:] = torch.softmax(vecs_13[..., 5:], dim=1)
        boxes_13 = get_boxes(idxs_13, vecs_13, 32, resize_h, cfg3.ANCHORS_GROUP[13])

        idxs_26, vecs_26 = get_offsets(output_26, 0.1)
        vecs_26[..., 5:] = torch.softmax(vecs_26[..., 5:], dim=1)
        boxes_26 = get_boxes(idxs_26, vecs_26, 16, resize_h, cfg3.ANCHORS_GROUP[26])

        idxs_52, vecs_52 = get_offsets(output_52, 0.1)
        vecs_52[..., 5:] = torch.softmax(vecs_52[..., 5:], dim=1)
        boxes_52 = get_boxes(idxs_52, vecs_52, 8, resize_h, cfg3.ANCHORS_GROUP[52])


        box_13 = box_change(boxes_13)
        box_26 = box_change(boxes_26)
        box_52 = box_change(boxes_52)
        # print(box_13.shape,box_26.shape,box_52.shape)
        # box_13 = nms(box_13,0.8)
        # box_26 = nms(box_26, 0.8)
        # box_52 = nms(box_52, 0.8)
        box = torch.cat([box_13, box_26, box_52], dim=0)
        if box.shape[0] != 0:
            box = nms(box, 0.1)
        print(box)
        for box_ in box:
            x1, y1, x2, y2, cls, cls_index = int(box_[0]), int(box_[1]), int(box_[2]), int(box_[3]), box_[4], int(box_[5])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.putText(image, cls_names[cls_index][:-1], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255))
        cv2.imshow("img", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
