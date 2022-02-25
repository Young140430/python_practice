import os
import xml
from xml.dom import minidom

# 存放xml文件的文件夹路径
xml_directory = r"E:\yolov3-output3\outputs"
# txt文件
txt = open(r"E:\yolov3_train3_img\label.txt", "w")
for i in os.listdir(xml_directory):
    # 读取xml文件
    xml_file = minidom.parse(os.path.join(xml_directory, i))
    # 获得xml文件中的根节点
    root = xml_file.documentElement
    # 按照名称查找子节点，获得需要得元素内容
    img_filename = root.getElementsByTagName("path")[0].firstChild.data[:]
    DOMTree = xml.dom.minidom.parse(
        os.path.join(xml_directory,
                     i.split(".")[0] + ".xml"))
    collection = DOMTree.documentElement
    boundingbox = collection.getElementsByTagName("item")
    for j in boundingbox:
        cls = j.getElementsByTagName("name")[0].childNodes[0].data
        if cls=="bird":
            cls2=0
        elif cls=="bull":
            cls2=1
        else:
            cls2=2
        xmin = j.getElementsByTagName("xmin")[0].firstChild.data
        xmax = j.getElementsByTagName("xmax")[0].firstChild.data
        ymin = j.getElementsByTagName("ymin")[0].firstChild.data
        ymax = j.getElementsByTagName("ymax")[0].firstChild.data
        w = int(xmax)-int(xmin)
        h = int(ymax)-int(ymin)
        cx = int(xmin) + w//2
        cy = int(ymin) + h//2
        # 给定txt文件每行写入格式
        label = f"{img_filename} {cls2} {cx} {cy} {w} {h}\n"
        txt.write(label)
txt.close()
