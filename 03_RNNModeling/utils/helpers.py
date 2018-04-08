import random
import string
import jieba
import os


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def read_vocab(file_path):
    with open(file_path, 'r') as fin:
        vocab = [x.strip() for x in fin.readlines()]
        vocab = [x.decode('utf-8') for x in vocab if x != '']
    return vocab


def custom_tokenizer(text):
    return [token for token in jieba.cut(text) if token != '\n']  # keep punctuations in the lyrics
    # if token not in string.punctuation+" " and token != '\n'


def read_data(filename, vocab, model, window, overlap):
    lines = [line.strip() for line in open(filename, 'r').readlines()]
    lines = [custom_tokenizer(text) for text in lines]
    while True:
        random.shuffle(lines)

        for text in lines:
            ori_text = vocab_encode(text, vocab)
            text = word2vec_lookup(text, model)
            for start in range(0, len(text) - window, overlap):
                tmp1 = ori_text[start: start + window]
                tmp2 = text[start: start + window]
                tmp1 += [0] * (window - len(tmp1))
                tmp2 += [[0.] * 300] * (window - len(tmp2))  # 300 dimensional word2vec
                yield (tmp1, tmp2)


def read_batch(stream, batch_size):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch


def word2vec_lookup(text, model):
    return [model.get_word_vector(word) for word in text]


def vocab_encode(text, vocab):
    return [vocab.index(x) + 1 for x in text if x in vocab]


def vocab_decode(array, vocab):
    return ''.join([vocab[x - 1] for x in array])
