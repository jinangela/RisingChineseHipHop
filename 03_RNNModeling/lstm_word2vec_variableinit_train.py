# -*- coding: utf-8 -*-
"""
Initial state non-zero initialization: https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
Beam Search: https://github.com/hunkim/word-rnn-tensorflow/blob/master/model.py
"""
import tensorflow as tf
import time
import os
import sys
from fastText import load_model
from utils.rnn_variable_init_state_utils import *
from utils.helpers import *


class CWordRNN(object):
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

    def create_rnn(self, seq, state_initializer):
        layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        batch = tf.shape(seq)[0]  # need to get the batch size from placeholder to make sure the initializer can work!
        # zero_states = cells.zero_state(batch, dtype=tf.float32)
        # self.in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]])   # shape: [batch_size, s]
        #                        for state in zero_states])                                   # for s in cell.state_size

        deterministic = tf.constant(False)
        if state_initializer == StateInitializer.ZERO_STATE:
            initializer = zero_state_initializer
        elif state_initializer == StateInitializer.VARIABLE_STATE:
            initializer = make_variable_state_initializer()
        elif state_initializer == StateInitializer.NOISY_ZERO_STATE:
            initializer = make_gaussian_state_initializer(zero_state_initializer, deterministic)
        elif state_initializer == StateInitializer.NOISY_VARIABLE_STATE:
            initializer = make_gaussian_state_initializer(make_variable_state_initializer(), deterministic)
        else:
            sys.exit(1)
        self.in_state = get_initial_cell_state(cells, initializer, batch, tf.float32)

        # this line to calculate the real length of seq
        # all seq are padded to be of the same length, which is num_steps
        # length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)
        length = tf.reduce_sum(tf.sign(tf.norm(seq, axis=2)), 1)
        self.output, self.out_state = tf.nn.dynamic_rnn(cells, seq, length, self.in_state)
        # output shape: [batch_size, max_time, state_size] = [32, 50, 256] in this case

    def create_model(self, state_initializer):
        seq = tf.one_hot(self.ori_seq-1, len(self.vocab))
        # vocab_encode has plus 1 but one_hot index should start from 0
        # Tried tf.map_fn(lambda x: self.model.get_word_vector(x), self.seq)
        # Got TypeError: getWordVector(): incompatible function arguments. The following argument types are supported:
        # 1. (self: fasttext_pybind.fasttext, arg0: fasttext_pybind.Vector, arg1: unicode) -> None
        self.create_rnn(self.seq_vec, state_initializer)  # used word2vec embeddings
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

            writer = tf.summary.FileWriter(os.path.join(self.root_dir, '03_RNNModeling/checkpoints_word2vec_varinit/summaries/'))
            writer.add_graph(sess.graph)

            ckpt_path = os.path.join(self.root_dir, '03_RNNModeling/checkpoints_word2vec_varinit/hiphop_generator')
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_path))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            stream = read_data(self.path, self.vocab, self.model, self.num_steps, overlap=self.num_steps // 2)
            data = read_batch(stream, self.batch_size)

            iteration = self.gstep.eval()
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
    safe_mkdir(os.path.join(root_dir, '03_RNNModeling/checkpoints_word2vec_varinit'))

    lm = CWordRNN(root_dir)
    lm.create_model(StateInitializer.VARIABLE_STATE)
    lm.train()


if __name__ == '__main__':
    main()
