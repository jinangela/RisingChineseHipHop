import urllib2
from bs4 import BeautifulSoup

webpage = 'https://www.ybdu.com/xiaoshuo/6/6376/'
html = urllib2.urlopen(webpage)
soup = BeautifulSoup(html, 'html.parser')
menu = soup.find_all('ul', {'class': 'mulu_list'})[0]

chapters = []
for link in menu.find_all('a'):
    chapters.append(str(link.get('href')))

print(len(chapters))

chapters = [webpage + item for item in chapters]
