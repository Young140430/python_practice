import torch
x = torch.Tensor([[204, 104, 248, 148, 0.93],
                  [194, 88, 255, 152, 0.80],
                  [169, 63, 269, 155, 0.73],
                  [478, 95, 523, 147, 0.9],
                  [466, 74, 530, 150, 0.82],
                  [331, 88, 373, 137, 0.95]])
print(x.shape)
print(x)
print(torch.argsort(x[:, 4], descending=True))
x = x[torch.argsort(x[:, 4], descending=True)]
print(x)
for e in range(x.shape[0]):
    if e == len(x):
        break
    else:
        x1 = x[:, 4].clone()
        x1[:] = 0
        a = e
        for i in range(e, len(x)):
            area1 = (x[e, 2] - x[e, 0]) * (x[e, 3] - x[e, 1])
            area2 = (x[i, 2] - x[i, 0]) * (x[i, 3] - x[i, 1])
            xj1 = torch.maximum(x[e, 0], x[i, 0])
            yj1 = torch.maximum(x[e, 1], x[i, 1])
            xj2 = torch.minimum(x[e, 2], x[i, 2])
            yj2 = torch.minimum(x[e, 3], x[i, 3])
            j_area = (xj2 - xj1) * (yj2 - yj1)
            iou = j_area / (area1 + area2 - j_area)
            x1[a] = iou
            a = a + 1
        is_one = torch.lt(x1, 0.2)
        is_one[e] = True
        x = x[is_one]
print(x)