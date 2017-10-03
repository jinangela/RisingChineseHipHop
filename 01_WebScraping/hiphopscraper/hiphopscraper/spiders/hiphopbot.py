# -*- coding: utf-8 -*-
import scrapy
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from collections import OrderedDict
from hiphopscraper.items import HiphopscraperItem


class HiphopbotSpider(scrapy.Spider):
    name = 'hiphopbot'
    allowed_domains = ['www.kuwo.cn']
    start_urls = ['http://www.kuwo.cn/album/3648470?catalog=yueku2016',
				  'http://www.kuwo.cn/album/3656723?catalog=yueku2016',
				  'http://www.kuwo.cn/album/3666150?catalog=yueku2016',
				  'http://www.kuwo.cn/album/3671934?catalog=yueku2016',
				  'http://www.kuwo.cn/album/3676089?catalog=yueku2016',
				  'http://www.kuwo.cn/album/3743211?catalog=yueku2016',
				  'http://www.kuwo.cn/album/3789374?catalog=yueku2016',
				  'http://www.kuwo.cn/album/3828324?catalog=yueku2016',
				  'http://www.kuwo.cn/album/3984153?catalog=yueku2016',
				  'http://www.kuwo.cn/album/3993906?catalog=yueku2016',
				  'http://www.kuwo.cn/album/4000972?catalog=yueku2016',
                  'http://www.kuwo.cn/album/4006884?catalog=yueku2016']

    def parse(self, response):
        for url in list(OrderedDict.fromkeys(response.xpath("//a[contains(@href, '/yinyue/')]/@href").extract())):
            yield scrapy.Request(url, callback=self.parse_lyrics, dont_filter=True)
    
    def parse_lyrics(self, response):
        item = HiphopscraperItem()
        item["title"] = response.xpath("//div[@id='musiclrc']/p/text()").extract()
        item["singer"] = response.xpath("//p[@class='artist']/span/a/text()").extract()
        item["episode"] = response.xpath("//p[@class='album']/span/a/text()").extract()
        item["lyrics"] = ";".join(response.xpath("//div[@id='llrcId']/p/text()").extract())
        yield item