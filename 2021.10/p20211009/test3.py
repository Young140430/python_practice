'''def pr(name,age):
    print("姓名：",name)
    print("年龄：",age)
pr(age=20,name="young")'''
def add(a,*b):
    sum=0
    for i in b:
        sum+=i
    print(a+sum)
#add(1,3,5,7,9,11,13,15,17,19)