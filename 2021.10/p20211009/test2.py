def changelist(mylist):
    mylist.append([1,2,3,4,5])
list=[10,20,30,40,50]
print("调用函数前：",list)
changelist(list)
print("调用函数后：",list)
dict1={}
def changedict(mydict):
    mydict["姓名"]="young"
print("调用函数前：",dict1)
changedict(dict1)
print("调用函数后：",dict1)