# Web Scraping
We use [scrapy](https://scrapy.org/) to scrape training data for the language model in [03_RNNModeling](https://github.com/jinangela/RisingChineseHipHop/tree/master/03_RNNModeling).

## Training Data Brief Intro
1. 152 rap lyrics from [中国有嘻哈 The Rap of China](https://en.wikipedia.org/wiki/The_Rap_of_China)
2. 160 chapters from [红楼梦 Dream of the Red Chamber](https://en.wikipedia.org/wiki/Dream_of_the_Red_Chamber)

## Getting Started
### Prerequisites
You need to install scrapy first: `pip install scrapy`.
Please see the [installation guide](https://doc.scrapy.org/en/latest/intro/install.html) if you have any questions or you prefer to use conda to install.

### Running the Spider
We followed the instructions in the scrapy [tutorial](https://doc.scrapy.org/en/latest/intro/tutorial.html) and created two Spiders, [hiphopscraper](https://github.com/jinangela/RisingChineseHipHop/tree/master/01_WebScraping/hiphopscraper) and [hongloumeng](https://github.com/jinangela/RisingChineseHipHop/tree/master/01_WebScraping/hongloumeng), to scrape rap lyrics from [酷我音乐 Kuwo Music](www.kuwo.cn) and book chapters from [笔趣阁全本小说网 ybdu.com](www.ybdu.com).

To run hiphopscraper, go to hiphopscraper's top level directory(there are detailed explanations on the folder structures [here](https://doc.scrapy.org/en/latest/intro/tutorial.html#creating-a-project) to help you if you are not sure what is "top level directory"), and run the following command in your command line prompt:    
```
scrapy crawl hiphopbot
```
where hiphopbot is the name of the spider and is defined in [hiphopbot.py](https://github.com/jinangela/RisingChineseHipHop/blob/master/01_WebScraping/hiphopscraper/hiphopscraper/spiders/hiphopbot.py).

Similarly, to run the spider for hongloumeng, run the following command:
```
scrapy crawl honglou
```

## Future Improvements
We may create another spider to crawl Chinese ancient poems later.
