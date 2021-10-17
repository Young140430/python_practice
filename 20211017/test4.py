import urllib.request
import urllib.parse
import re
import os
header= \
{
    'User-Agent':'Mozilla (Windows 10.0; Win64; x64) Chrome/56.0.2924.87',
    'referer':'https://image.baidu.com'
    }
url="https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=index&fr=&hs=0&xthttps=111110&sf=1&fmq=&pv={pageNum}&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word={word}&oq={word}&rsp=-1"
keyword=input("请输入搜索关键字：")
keyword=urllib.parse.quote(keyword,"utf-8")
n=0
j=0
while(n<3000):
    n+=30
    url1=url.format(word=keyword,pageNum=str(n))
    rep=urllib.request.Request(url1,headers=header)
    rep=urllib.request.urlopen(rep)
    print(rep.read().decode("utf-8"))