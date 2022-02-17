import os
import xml
from xml.dom import minidom

voc_cls = ['person','bird','cat','cow','dog','horse','sheep','aeroplane','bicycle','boat','bus','car','motorbike',
           'train','bottle','chair','diningtable','pottedplant','sofa','tvmonitor']
train_cls=['bird','bull','sheep']
# 存放xml文件的文件夹路径
xml_directory = r"E:\swimming-pool-and-car\training_data\training_data\labels"
# txt文件
txt = open(r"E:\swimming-pool-and-car\training_data\training_data\label.txt", "w")
for i in os.listdir(xml_directory):
    # 读取xml文件
    xml_file = minidom.parse(os.path.join(xml_directory, i))
    # 获得xml文件中的根节点
    root = xml_file.documentElement
    # 按照名称查找子节点，获得需要得元素内容
    img_filename = root.getElementsByTagName("filename")[0].firstChild.data
    txt.write(f"{img_filename} ")
    img_w = root.getElementsByTagName("width")[0].firstChild.data
    img_h = root.getElementsByTagName("height")[0].firstChild.data
    if img_h != img_w:
        img_l = max(float(img_w),float(img_h))
    else:
        img_l = float(img_h)
    w_rate = float(img_l)/416
    h_rate = float(img_l)/416
    DOMTree = xml.dom.minidom.parse(
        os.path.join(xml_directory,
                     i.split(".")[0] + ".xml"))
    collection = DOMTree.documentElement
    boundingbox = collection.getElementsByTagName("object")
    for j in boundingbox:
        cls = j.getElementsByTagName("name")[0].childNodes[0].data
        # for k in range(3):
        #     if train_cls[k] == cls:
        #         cls2 = k
        cls2 = int(cls)-1
        xmin = j.getElementsByTagName("xmin")[0].firstChild.data
        xmax = j.getElementsByTagName("xmax")[0].firstChild.data
        ymin = j.getElementsByTagName("ymin")[0].firstChild.data
        ymax = j.getElementsByTagName("ymax")[0].firstChild.data
        w = float(xmax)-float(xmin)
        h = float(ymax)-float(ymin)
        cx = float(xmin) + w/2
        cy = float(ymin) + h/2
        w = int(w/w_rate)
        h = int(h/h_rate)
        cx = int(cx/w_rate)
        cy = int(cy/h_rate)
        # 给定txt文件每行写入格式
        label = f"{cls2} {cx} {cy} {w} {h} "
        txt.write(label)
    txt.write("\n")
txt.close()
