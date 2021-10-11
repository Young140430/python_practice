s=input("请输入字符串：")
s2=''
for i in range(len(s)):
    s1=s[len(s)-i-1]
    s2+=s1
print(s2)