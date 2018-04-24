# RNN Modeling on Rap Lyrics
A Chinese Rap Lyrics Generator.    
For learning Recurrent Neural Networks.    
Also for fun!

## Getting Started
### Prerequisites
You need to install the following python packages before running any models:
* [tensorflow](https://www.tensorflow.org/install/)
* [jieba](https://pypi.org/project/jieba/)
* [fastText](https://github.com/facebookresearch/fastText/tree/master/python)    
  - [fastText](https://fasttext.cc/) is a library for efficient learning of word representations and sentence classification.
  - [facebookresearch](https://github.com/facebookresearch) has published pre-trained word vectors(both pre-trained models trained on Wikipedia data using fastText and word embedding matrices) for 294 languages, you can find them [here](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md).
  - We use fastText for Chinese word embedding lookup.

### Model Architecture
[Andrej Karpathy](https://cs.stanford.edu/people/karpathy/) has a blog that demonstrates "[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)".

## Acknowledgement
* [lstm_train.py](https://github.com/jinangela/RisingChineseHipHop/blob/master/03_RNNModeling/lstm_train.py) adapted from Chip Huyen's example [11_char_rnn.py](https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/11_char_rnn.py) - "a clean, no-frills character-level generative language model".
