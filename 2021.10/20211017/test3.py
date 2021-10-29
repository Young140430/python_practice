from urllib.request import urlopen
from bs4 import BeautifulSoup
html=urlopen("http://www.shuhai.com/shuku/0_0_0_0_0_0_1_1.html")\
    .read().decode("gbk")
#print(html)
soup=BeautifulSoup(html,features="lxml")
all_a=soup.find_all('a')
print(all_a)
for i in all_a:
    print(i.get_text())