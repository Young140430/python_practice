from xml.dom.minidom import parse
import xml.dom.minidom
import os
root_path=r"E:/swimming-pool-and-car"
imgs_name = os.listdir(os.path.join(root_path, "training_data\\training_data\\images"))
# acquire bounding box
imgs_lable_dict = {}
for img_name in imgs_name:
    DOMTree = xml.dom.minidom.parse(
        os.path.join(os.path.join(root_path, "training_data\\training_data\\labels"),
                     img_name.split(".")[0] + ".xml"))
    collection = DOMTree.documentElement
    boundingbox = collection.getElementsByTagName("object")
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
print(imgs_lable_dict)