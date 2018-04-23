# Simple Word Cloud of Rap Lyrics from *The Rap of China*
## Getting Started
### Prerequisites
You need to install the following python packages before running the jupyter notebook [EDA.ipynb](https://github.com/jinangela/RisingChineseHipHop/blob/master/02_TextExploratoryAnalysis/EDA.ipynb):
* [jieba](https://pypi.org/project/jieba/)    
“结巴”中文分词：做最好的 Python 中文分词组件    
“Jieba” (Chinese for “to stutter”) Chinese text segmentation: built to be the best Python Chinese word segmentation module.
* [nltk](https://www.nltk.org/install.html)
* [wordcloud](https://pypi.org/project/wordcloud/)
* [chardet](https://pypi.org/project/chardet/)

### Data Cleaning & Preprocessing
**Step 0**: We removed all the informational words(with the colons) before tokenization, including but not limited to:
1. 演唱：
2. 词：
3. 曲：
4. 编曲：
5. 定位制作人：
6. 音乐总监：

We used regex `u'\xef\xbc\x9a'` to search for Chinese colons and removed all the sentences that contain them.

**Step 1**: 分词 Tokenization    
Thanks to [jieba](https://github.com/fxsjy/jieba), Chinese tokenization is as easy as `jieba.cut(text)`.

**Step 2**: 去除停用词&统计词频 Removing Stopwords & Frequency Count    
**TODO**: Use this Chinese stopword list instead: https://github.com/stopwords-iso/stopwords-zh/blob/master/stopwords-zh.txt    
We removed both Chinese and English stopwords.
* The current Chinese stopword list comes from https://gist.github.com/dreampuf/5548203 and it will be replaced by https://github.com/stopwords-iso/stopwords-zh/blob/master/stopwords-zh.txt since the latter has a licence to ensure the quality.
* The English stopword list comes from `nltk.corpus.stopwords.words('english')`.

**Step 3**: 制作词云图 Create Word Cloud    
Use the [wordcloud](https://pypi.org/project/wordcloud/) package to create a `WordCloud` object and feed frequency counts of words into it to create a word cloud plot using [matplotlib](https://matplotlib.org/) package.    
Example word cloud plot:
![word cloud example](https://github.com/jinangela/RisingChineseHipHop/blob/master/02_TextExploratoryAnalysis/Episode%2012%20Word%20Cloud%20Test.png)

## Acknowledgements
Inspired by 大数据文摘 | bigdatadigest <<[十分钟视频,手把手教你用Python撒情人节狗粮的正确姿势](http://mp.weixin.qq.com/s/ux2MqsjUwalHiIsm1f832w)>>.
