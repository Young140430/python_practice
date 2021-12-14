from xml.dom.minidom import parse
import xml.dom.minidom
import os
root_path=r"E:/3"
imgs_name = os.listdir(root_path)
# acquire bounding box
imgs_lable_dict = {}
for img_name in imgs_name:
    DOMTree = xml.dom.minidom.parse(
        os.path.join(r"E:/outputs",
                     img_name.split(".")[0] + ".xml"))
    collection = DOMTree.documentElement
    boundingbox = collection.getElementsByTagName("item")
    imgs_lable_dict[img_name] = []
    for i in boundingbox:
        category = i.getElementsByTagName("name")[0].childNodes[0].data
        tmp=[]
        tmp.append(float(
            [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmin")][
                0]))
        tmp.append(float(
            [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymin")][
                0]))
        tmp.append(float(
            [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmax")][
                0]))
        tmp.append(float(
            [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymax")][
                0]))
        tmp.append(category)
        imgs_lable_dict[img_name].append(tmp)
for i in range(22001,33000):
    if imgs_lable_dict['0'+str(i)+'.jpg']==[]:
        print(i)