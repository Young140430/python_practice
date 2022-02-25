#
# x="Please enter a text: \n"
# y = '123456789784515'
#
#
#
# f = open('Output.txt', 'w')
#
# f.write(x)
# f.write(y)
#
# f.close()
import numpy as np
import torch
import cv2

a = torch.tensor([[0.5, 0.9],
                  [0.3, 0.1],
                  [0.9, 0.8]])
b = torch.ge(a, 0.5).float()
print(b)
b = np.array(b,dtype = np.uint8)
cv2.im
# b = np.nonzero(b)
# print(b)
