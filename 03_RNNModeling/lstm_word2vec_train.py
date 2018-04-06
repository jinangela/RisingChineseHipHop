# -*- coding: utf-8 -*-
"""
Initial state non-zero initialization: https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
Beam Search: https://github.com/hunkim/word-rnn-tensorflow/blob/master/model.py
"""
import tensorflow as tf
import time
import os
import random
import jieba
import string
from fastText import load_model


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


class CharRNN(object):
    def __init__(self, root_dir):
        """
        A Chinese Hip Hop Song Lyrics Generator
        """
        self.root_dir = root_dir
        self.path = os.path.join(root_dir, '01_WebScraping/lyrics.txt')
        self.vocab = read_vocab(os.path.join(root_dir, '01_WebScraping/vocab.txt'))
        self.model = load_model(os.path.join(root_dir, '03_RNNModeling/zh/wiki.zh.bin'))
        self.ori_seq = tf.placeholder(tf.int32, [None, None])
        self.seq_vec = tf.placeholder(tf.float32, [None, None, 300])  # 300 dimensional word2vec
        # self.temp = tf.constant(1.5)
        self.hidden_sizes = [128, 256]
        self.batch_size = 32
        self.lr = 0.0003
        self.skip_step = 1
        self.num_steps = 50  # for RNN unrolled
        self.len_generated = 200
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def create_rnn(self, seq):
        layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        batch = tf.shape(seq)[0]
        zero_states = cells.zero_state(batch, dtype=tf.float32)
        self.in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]])   # shape: [batch_size, s]
                               for state in zero_states])                                   # for s in cell.state_size
        # this line to calculate the real length of seq
        # all seq are padded to be of the same length, which is num_steps
        # length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)
        length = tf.reduce_sum(tf.sign(tf.norm(seq, axis=2)), 1)
        self.output, self.out_state = tf.nn.dynamic_rnn(cells, seq, length, self.in_state)
        # output shape: [batch_size, max_time, state_size] = [32, 50, 256] in this case

    def create_model(self):
        seq = tf.one_hot(self.ori_seq-1, len(self.vocab))
        # vocab_encode has plus 1 but one_hot index should start from 0
        # Tried tf.map_fn(lambda x: self.model.get_word_vector(x), self.seq)
        # Got TypeError: getWordVector(): incompatible function arguments. The following argument types are supported:
        # 1. (self: fasttext_pybind.fasttext, arg0: fasttext_pybind.Vector, arg1: unicode) -> None
        self.create_rnn(self.seq_vec)  # used word2vec embeddings
        self.logits = tf.layers.dense(self.output, len(self.vocab), None)  # activation is None so only Wx + b here

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:, :-1],
                                                       labels=seq[:, 1:])
        # in self.logits we use ":-1" because the model is generating probabilities for the next word
        self.loss = tf.reduce_mean(loss)  # Why reduce_sum???
        # sample the next character from Maxwell-Boltzmann Distribution
        # with temperature temp. It works equally well without tf.exp
        # self.sample = tf.multinomial(tf.exp(self.logits[:, -1] / self.temp), 1)[:, 0]
        self.sample = tf.multinomial(self.logits[:, -1], 1)[:, 0]
        # it's not a good idea to sample one word from a large vocabulary
        # TODO: Add Beam Search
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)
        self.l = tf.summary.scalar("loss", self.loss)

    def train(self):
        saver = tf.train.Saver()
        start = time.time()
        min_loss = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(os.path.join(self.root_dir, '03_RNNModeling/summaries/'))
            writer.add_graph(sess.graph)

            ckpt_path = os.path.join(self.root_dir, '03_RNNModeling/checkpoints_word2vec/hiphop_generator')
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_path))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            iteration = self.gstep.eval()
            stream = read_data(self.path, self.vocab, self.model, self.num_steps, overlap=self.num_steps // 2)
            data = read_batch(stream, self.batch_size)
            while True:
                ori_batch, batch = zip(*next(data))

                # for batch in read_batch(read_data(DATA_PATH, vocab)):
                batch_loss, _, batch_summary = sess.run([self.loss, self.opt, self.l],
                                                        {self.ori_seq: ori_batch, self.seq_vec: batch})
                if (iteration + 1) % self.skip_step == 0:
                    print('Iter {}. \n    Loss {}. Time {}'.format(iteration, batch_loss, time.time() - start))
                    writer.add_summary(batch_summary, iteration)
                    self.online_infer(sess)
                    start = time.time()
                    if min_loss is None:
                        saver.save(sess, ckpt_path, iteration)
                    elif batch_loss < min_loss:
                        saver.save(sess, ckpt_path, iteration)
                        min_loss = batch_loss
                iteration += 1

    def online_infer(self, sess):
        """ Generate sequence one character at a time, based on the previous character
        """
        for seed in [u'\u5982\u679c\u8bf4', u'\u8fd9\u4e2a']:
            sentence = seed
            state = None
            for _ in range(self.len_generated):
                batch = [word2vec_lookup(sentence[0], self.model)]
                feed = {self.seq_vec: batch}
                if state is not None:  # for the first decoder step, the state is None
                    for i in range(len(state)):
                        feed.update({self.in_state[i]: state[i]})
                index, state = sess.run([self.sample, self.out_state], feed)
                sentence += vocab_decode(index+1, self.vocab)  # +1?
            print('\t' + sentence)


def main():
    root_dir = '/Users/jinangela/Documents/IndependentResearch/RisingChineseHipHop/'
    safe_mkdir(os.path.join(root_dir, '03_RNNModeling/checkpoints_word2vec'))

    lm = CharRNN(root_dir)
    lm.create_model()
    lm.train()


if __name__ == '__main__':
    main()
