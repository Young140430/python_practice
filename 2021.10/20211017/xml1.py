import xml.etree.cElementTree as et
tree=et.parse("1.xml")
root=tree.getroot()
#print(root,root.findall("object"))
for o in root.findall("object"):
    name=o.find("name").text
    print(name)
    for b in o.findall("bndbox"):
        xmin = b.find("xmin").text
        ymin = b.find("ymin").text
        xmax = b.find("xmax").text
        ymax = b.find("ymax").text
        print(xmin)
        print(ymin)
        print(xmax)
        print(ymax)
    print("*************************************")

