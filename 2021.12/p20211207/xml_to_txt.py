import os
from xml.dom import minidom

# 存放xml文件的文件夹路径
xml_directory = r"E:\outputs"
# txt文件
txt = open(r"E:\xml_to_txt.txt", "w")
for i in os.listdir(xml_directory):
    # 读取xml文件
    xml_file = minidom.parse(os.path.join(xml_directory, i))
    # 获得xml文件中的根节点
    root = xml_file.documentElement
    # 按照名称查找子节点，获得需要得元素内容
    img_filename = root.getElementsByTagName("path")[0].firstChild.data[-10:]
    xmin = root.getElementsByTagName("xmin")[0].firstChild.data
    xmax = root.getElementsByTagName("xmax")[0].firstChild.data
    ymin = root.getElementsByTagName("ymin")[0].firstChild.data
    ymax = root.getElementsByTagName("ymax")[0].firstChild.data
    # 给定txt文件每行写入格式
    label = f"{img_filename} {xmin} {ymin} {xmax} {ymax}\n"
    txt.write(label)
txt.close()
