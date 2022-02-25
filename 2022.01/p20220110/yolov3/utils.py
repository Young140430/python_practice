import torch
from torch import long, argsort


def iou(box, boxes, mode="inter"):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])

    w = torch.clamp(x2 - x1, min=0)
    h = torch.clamp(y2 - y1, min=0)

    inter = w * h

    if mode == 'inter':
        return inter / (box_area + boxes_area - inter)
    elif mode == 'min':
        return inter / torch.min(box_area, boxes_area)


def nms(boxes, thresh, mode='inter'):
    args = boxes[:, 4].argsort(descending=True)
    sort_boxes = boxes[args]
    keep_boxes = []

    while len(sort_boxes) > 0:
        _box = sort_boxes[0]
        keep_boxes.append(_box)

        if len(sort_boxes) > 1:
            _boxes = sort_boxes[1:]
            _iou = iou(_box, _boxes, mode)
            sort_boxes = _boxes[_iou < thresh]
        else:
            break

    return keep_boxes


# def detect(feature_map, thresh):
#     masks = feature_map[:, 4, :, :] > thresh
#     idxs = torch.nonzero(masks)


if __name__ == '__main__':
    box = torch.Tensor([2, 2, 3, 3, 6])
    boxes = torch.Tensor([[2, 2, 3, 3, 6], [2, 2, 4, 4, 5], [2, 2, 5, 5, 4]])
    print(iou(box, boxes, mode="inter"))
    print(nms(boxes, 0.1))
    # import numpy as np
    #
    # a = np.array([[1, 2], [3, 4]])
    # print(a[:, 1])
