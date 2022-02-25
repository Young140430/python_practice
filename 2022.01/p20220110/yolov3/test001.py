import glob
import xml.etree.cElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = r"E:\swimming-pool-and-car\training_data\training_data\labels"
CLUSTERS = 6


def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        try:
            for obj in tree.iter("object"):
                xmin = float(obj.findtext("bndbox/xmin")) / width
                ymin = float(obj.findtext("bndbox/ymin")) / height
                xmax = float(obj.findtext("bndbox/xmax")) / width
                ymax = float(obj.findtext("bndbox/ymax")) / height

                xmin = np.float64(xmin)
                ymin = np.float64(ymin)
                xmax = np.float64(xmax)
                ymax = np.float64(ymax)
                if xmax == xmin or ymax == ymin:
                    print(xml_file)
                dataset.append([xmax - xmin, ymax - ymin])
        except:
            print(xml_file)
    return np.array(dataset)


if __name__ == '__main__':
    # print(__file__)
    data = load_dataset(ANNOTATIONS_PATH)
    out = kmeans(data, k=CLUSTERS)
    # clusters = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
    # out= np.array(clusters)/416.0
    print(out)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}-{}".format(out[:, 0] * 416, out[:, 1] * 416))

    ratios = np.around(out[:, 0] * out[:, 1] * 416 * 416, decimals=2).tolist()
    print("Ratios:\n {}".format(ratios))
    print("Sorted Ratios:\n {}".format(sorted(ratios)))
    '''[[0.07       0.192     ]
        [0.23       0.232     ]
        [0.13       0.37066667]
        [0.562      0.41236702]
        [0.436      0.75975976]
        [0.038      0.06666667]
        [0.25       0.54666667]
        [0.118      0.1021021 ]
        [0.842      0.864     ]]
        
        [[0.04959821 0.04959821]
        [0.03580357 0.00763393]
        [0.04959821 0.04959821]
        [0.04959821 0.04964286]
        [0.00919643 0.039375  ]
        [0.19839286 0.19839286]]'''
    '''[[29.12 79.872]
        [95.68 96.512]
        [54.08 154.1973]
        [233.792 171.5447]
        [181.376 316.0601]
        [15.808 27.7333]
        [104 227.4133]
        [49.088 42.4745]
        [350.272 359.424]]
        
        [[83 83]
        [21 21]
        [21 20]
        [19 22]
        [4 16]
        [15 3]]'''