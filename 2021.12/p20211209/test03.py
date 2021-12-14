import numpy as np

boxes = np.array([[1,3,4,5,6],[1,2,3,1,4],[4,2,1,3,2],[2,2,1,1,3]])
print((-boxes[:,4]).argsort())
print(boxes[(-boxes[:,4]).argsort()])