num=int(input("请输入一个正整数："))
while num>0:
    a=num%10
    print(a,end="")
    num//=10