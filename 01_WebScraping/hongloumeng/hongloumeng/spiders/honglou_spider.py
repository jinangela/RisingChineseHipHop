"""Ref: https://doc.scrapy.org/en/latest/intro/tutorial.html"""
import os
import scrapy
import urllib2
from bs4 import BeautifulSoup


class HonglouSpider(scrapy.Spider):
    name = "honglou"
    allowed_domains = ['https://www.ybdu.com/xiaoshuo/6/6376/']  # menu of honglou

    # get start_urls
    html = urllib2.urlopen(allowed_domains[0])
    soup = BeautifulSoup(html, 'html.parser')
    menu = soup.find_all('ul', {'class': 'mulu_list'})[0]

    chapters = []
    for link in menu.find_all('a'):
        chapters.append(str(link.get('href')))
    chapters = [allowed_domains[0] + item for item in chapters]

    start_urls = chapters

    def parse(self, response):
        # title = response.xpath("//div[@class='h1title']/h1/text()").extract()[0]
        filename = response.url.split('/')[-1].split('.')[0]
        text = response.xpath("//div[@id='htmlContent']/text()").extract()
        text = [item.strip() for item in text if item.strip() != u'']

        with open(os.path.join('/Users/jinangela/Documents/IndependentResearch/RisingChineseHipHop/01_WebScraping/'
                               'hongloumeng/chapters', filename + '.txt'), 'w') as f:
            for item in text:
                f.write('%s\n' % item.encode('utf-8'))
