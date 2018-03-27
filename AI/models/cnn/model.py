import codecs

import numpy as np
import tensorflow as tf

from config import *
from .layers.ConvolutionalLayer import Convolutional1D
from .layers.DenseLayer import SoftmaxDense
from .layers.EmbeddingLayer import Embedding
from .utils.evaluate_util import sim_compute
from .utils.tensor_util import zero_nil_slot


class CNNModel(object):
    def __init__(
            self, max_len, word_weights, tag_weights, epoch_results_path,
            result_file=None, label_voc=None):
        """
        Initialize model
        Args:
            max_len: int, 句子最大长度
            word_weights: np.array, shape=[|V_words|, w2v_dim]，词向量
            tag_weights: np.array, shape=[|V_tags|, t2v_dim],标记向量
            result_file: str, 模型评价结果存放路径
            epoch_results_path: str, 每个epoch产生的文件存放路径
            label_voc: dict
        """
        self._result_file = result_file
        self._epoch_results_path = epoch_results_path
        self._label_voc = label_voc
        self._label_voc_rev = dict()
        self.nb_epoch_scores = []  # 存放nb_epoch次迭代的f值
        for key in self._label_voc:
            value = self._label_voc[key]
            self._label_voc_rev[value] = key

        # input placeholders
        self.input_sentence_ph = tf.placeholder(
            tf.int32, shape=(None, max_len), name='input_sentence_ph')
        self.input_tag_ph = tf.placeholder(tf.int32, shape=(None, max_len), name='input_tag_ph')
        self.label_ph = tf.placeholder(tf.int32, shape=(None,), name='label_ph')
        self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob')
        self.word_keep_prob_ph = tf.placeholder(tf.float32, name='word_keep_prob')
        self.tag_keep_prob_ph = tf.placeholder(tf.float32, name='tag_keep_prob')

        # embedding layers
        self.nil_vars = set()
        word_embed_layer = Embedding(
            params=word_weights, ids=self.input_sentence_ph,
            keep_prob=self.word_keep_prob_ph, name='word_embed_layer')
        tag_embed_layer = Embedding(
            params=tag_weights, ids=self.input_tag_ph,
            keep_prob=self.tag_keep_prob_ph, name='tag_embed_layer')
        self.nil_vars.add(word_embed_layer.params.name)
        self.nil_vars.add(tag_embed_layer.params.name)

        # sentence representation
        sentence_input = tf.concat(
            values=[word_embed_layer.output, tag_embed_layer.output], axis=2)

        # sentence conv
        conv_layer = Convolutional1D(
            input_data=sentence_input, filter_length=3,
            nb_filter=1000, activation='relu', name='conv_layer')

        # dense layer
        dense_input_drop = tf.nn.dropout(conv_layer.output, self.keep_prob_ph)
        self.dense_layer = SoftmaxDense(
            input_data=dense_input_drop, input_dim=conv_layer.output_dim,
            output_dim=NB_LABELS, name='output_layer')

        self.loss = self.dense_layer.loss(self.label_ph) + \
                    0.001 * tf.nn.l2_loss(self.dense_layer.weights)
        optimizer = tf.train.AdamOptimizer()  # Adam
        grads_and_vars = optimizer.compute_gradients(self.loss)
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self.nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # train op
        self.train_op = optimizer.apply_gradients(
            nil_grads_and_vars, name='train_op', global_step=global_step)

        # pre op
        self.pre_op = self.dense_layer.get_pre_y()

        # summary
        gpu_options = tf.GPUOptions(visible_device_list='0', allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # init model
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def fit(self, sentences_train, tags_train, labels_train,
            sentences_dev=None, tags_dev=None, labels_dev=None,
            sentences_test=None, tags_test=None, labels_test=None,
            batch_size=64, nb_epoch=40, keep_prob=1.0, word_keep_prob=1.0,
            tag_keep_prob=1.0, seed=137):
        """
        fit model
        :param sentences_train: 训练数据
        :param tags_train: 训练数据
        :param labels_train: 训练数据
        :param sentences_dev: 开发数据
        :param tags_dev: 开发数据
        :param labels_dev: 开发数据
        :param sentences_test:
        :param tags_test:
        :param labels_test:
        :param batch_size: int
        :param nb_epoch: int, 迭代次数
        :param keep_prob: float between [0, 1], 全连接层前的dropout
        :param word_keep_prob: float between [0, 1], 词向量层dropout
        :param tag_keep_prob: float between [0, 1], 标记向量层dropout
        :param seed:
        :return:
        """
        nb_train = int(labels_train.shape[0] / batch_size) + 1
        for step in range(nb_epoch):
            print('Epoch %d / %d:' % (step + 1, nb_epoch))
            # shuffle
            np.random.seed(seed)
            np.random.shuffle(sentences_train)
            np.random.seed(seed)
            np.random.shuffle(tags_train)
            np.random.seed(seed)
            np.random.shuffle(labels_train)

            # train
            total_loss = 0.
            for i in range(nb_train):
                # for i in range(nb_train):
                sentences_feed = sentences_train[i * batch_size:(i + 1) * batch_size]
                tags_feed = tags_train[i * batch_size:(i + 1) * batch_size]
                labels_feed = labels_train[i * batch_size:(i + 1) * batch_size]
                feed_dict = {
                    self.input_sentence_ph: sentences_feed,
                    self.input_tag_ph: tags_feed,
                    self.label_ph: labels_feed,
                    self.keep_prob_ph: keep_prob,
                    self.word_keep_prob_ph: word_keep_prob,
                    self.tag_keep_prob_ph: tag_keep_prob,
                }
                _, loss_value = self.sess.run(
                    [self.train_op, self.loss], feed_dict=feed_dict)
                total_loss += loss_value

            total_loss /= float(nb_train)

            #  计算在训练集、开发集、测试集上的性能
            p_train, r_train, f_train = self.evaluate(sentences_train, tags_train, labels_train)
            p_dev, r_dev, f_dev = self.evaluate(sentences_dev, tags_dev, labels_dev)
            pre_labels = self.predict(sentences_test, tags_test)

            with codecs.open(self._epoch_results_path + '%d.csv' % (step + 1), 'w', encoding='utf-8') as file_w:
                for num, label in enumerate(pre_labels):
                    file_w.write('%d,%s\n' % (num + 1, self._label_voc_rev[label]))
            self.nb_epoch_scores.append([p_dev, r_dev, f_dev])
            print('\tloss=%f, train f=%f, dev f=%f' % (total_loss, f_train, f_dev))

    def predict(self, data_sentences, data_tags, batch_size=50):
        """
        Args:
            data_sentences, data_tags: np.array
            batch_size: int
        Return:
            pre_labels: list
        """
        pre_labels = []
        nb_test = int(data_sentences.shape[0] / batch_size) + 1
        for i in range(nb_test):
            sentences_feed = data_sentences[i * batch_size:(i + 1) * batch_size]
            tags_feed = data_tags[i * batch_size:(i + 1) * batch_size]
            feed_dict = {
                self.input_sentence_ph: sentences_feed,
                self.input_tag_ph: tags_feed,
                self.keep_prob_ph: 1.0,
                self.word_keep_prob_ph: 1.0,
                self.tag_keep_prob_ph: 1.0}
            pre_temp = self.sess.run(self.pre_op, feed_dict=feed_dict)
            pre_labels += list(pre_temp)
        return pre_labels

    def evaluate(self, data_sentences, data_tags, data_labels,
                 ignore_label=None, batch_size=64, simple_compute=True):
        """
        Args:
            data_sentences, data_tags, data_labels: np.array
            ignore_label: int, 负例的编号，或者None
            simple_compute: bool, 是否画出性能详细指标表格
        Return:
            p, r, f1
        """
        pre_labels = []
        nb_dev = int(len(data_labels) / batch_size) + 1
        for i in range(nb_dev):
            sentences_feed = data_sentences[i * batch_size:(i + 1) * batch_size]
            tags_feed = data_tags[i * batch_size:(i + 1) * batch_size]
            labels_feed = data_labels[i * batch_size:(i + 1) * batch_size]
            feed_dict = {
                self.input_sentence_ph: sentences_feed,
                self.input_tag_ph: tags_feed,
                self.label_ph: labels_feed,
                self.keep_prob_ph: 1.0,
                self.word_keep_prob_ph: 1.0,
                self.tag_keep_prob_ph: 1.0}
            pre_temp = self.sess.run(self.pre_op, feed_dict=feed_dict)
            pre_labels += list(pre_temp)
        right_labels = data_labels[:len(pre_labels)]
        pre, rec, f = sim_compute(pre_labels, right_labels, ignore_label=ignore_label)
        return pre, rec, f

    def clear_model(self):
        tf.reset_default_graph()  #
        self.sess.close()

    def get_best_score(self):
        """
        计算模型得分(当开发集上f值达到最高时所对应的测试集得分)
        Returns:
            score: float, 开发集达到最高时,测试集的[p, r, f]
            nb_epoch: int, the num of epoch
        """
        # nb_epoch_scores = sorted(self.nb_epoch_scores, key=lambda d: d[1][-1], reverse=True)
        nb_epoch, best_score = -1, None
        for i in range(len(self.nb_epoch_scores)):
            if not best_score or self.nb_epoch_scores[i][-1] > best_score[-1]:
                best_score = self.nb_epoch_scores[i]
                nb_epoch = i
        return best_score, nb_epoch
