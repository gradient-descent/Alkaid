import os
import pickle

import numpy as np

from .config import SEPARATOR, WORD_VOC_START, TAG_VOC_START, W2V_DIM, TAG_DIM
from .data import create_dictionary
from .io import read_lines
from .preprocessor import get_labels_vocabulary


def init_vocabulary(account,
                    input_segment_file,
                    word_voc_file,
                    tag_voc_file,
                    label_voc_file,
                    regen=None):
    """

    :param account:
    :param input_segment_file:
    :param word_voc_file:
    :param tag_voc_file:
    :param label_voc_file:
    :param regen:
    :return:
    """
    # If using regen model and need not re-generate vocabulary files,directly jump out without any operation
    if regen and ('_voc' not in regen):
        return

    lines = read_lines(input_segment_file)
    words = []
    pos_tags = []
    for line in lines:
        try:
            label_text = line.split(SEPARATOR)
            sentence = label_text[1]
            # words and tags
            words_tags = sentence.split(' ')
            words_temp, tag_temp = [], []
            for item in words_tags:
                try:
                    r_index = item.rindex('/')
                    word, tag = item[:r_index], item[r_index + 1:]
                    words_temp.append(word)
                    tag_temp.append(tag)
                except ValueError:
                    print('Occurring null word or tag.')
            pos_tags.extend(tag_temp)
            words.extend(words_temp)

            # word voc
            if regen and regen != 'word_voc':  # If one of the three dictionaries need to be regenerate but not 'word'
                pass
            else:
                create_dictionary(
                    words, word_voc_file, start=WORD_VOC_START,
                    min_count=5, sort=True, lower=True, overwrite=True)
            # tag voc
            if regen and regen != 'tag_voc':
                pass
            else:
                create_dictionary(
                    pos_tags, tag_voc_file, start=TAG_VOC_START,
                    sort=True, lower=False, overwrite=True)
            # label voc
            if regen and regen != 'label_voc':
                pass
            else:
                label_types = [str(i) for i in get_labels_vocabulary(account)]
                create_dictionary(
                    label_types, label_voc_file, start=0, overwrite=True)

        except ValueError:
            print('Occurring null line.')


def _init_word_embedding(
        word_embedding_file,
        word2vec_file,
        word_voc_file,
        overwrite=False):
    """
    Init word embedding
    :param word_embedding_file:
    :param word2vec_file:
    :param word_voc_file:
    :param overwrite:
    :return:
    """
    if os.path.exists(word_embedding_file) and not overwrite:
        return
    with open(word2vec_file, 'rb') as file:
        w2v_dict_full = pickle.load(file)
    with open(word_voc_file, 'rb') as file:
        w2id_dict = pickle.load(file)
    word_voc_size = len(w2id_dict.keys()) + WORD_VOC_START
    word_weights = np.zeros((word_voc_size, W2V_DIM), dtype='float32')
    for word in w2id_dict:
        index = w2id_dict[word]
        if word in w2v_dict_full:
            word_weights[index, :] = w2v_dict_full[word]
        else:
            random_vec = np.random.uniform(
                -0.25, 0.25, size=(W2V_DIM,)).astype('float32')
            word_weights[index, :] = random_vec
    # 写入pkl文件
    with open(word_embedding_file, 'wb') as file:
        pickle.dump(word_weights, file, protocol=2)


def _init_tag_embedding(tag_embedding_file, tag_voc_file, overwrite=False):
    """
    Init part-of-speech tag embedding
    :param tag_embedding_file:
    :param tag_voc_file:
    :param overwrite:
    :return:
    """
    if os.path.exists(tag_embedding_file) and not overwrite:
        return
    with open(tag_voc_file, 'rb') as file:
        tag_voc = pickle.load(file)
    tag_voc_size = len(tag_voc.keys()) + TAG_VOC_START
    tag_weights = np.random.normal(
        size=(tag_voc_size, TAG_DIM)).astype('float32')
    for i in range(TAG_VOC_START):
        tag_weights[i, :] = 0.
    with open(tag_embedding_file, 'wb') as file:
        pickle.dump(tag_weights, file, protocol=2)


def init_embedding(
        embedding_root,
        word2vec_file,
        word_embedding_file,
        tag_embedding_file,
        word_voc_file,
        tag_voc_file,
        regen=None):
    """
    Init word and tag embedding.
    :param embedding_root:
    :param word2vec_file:
    :param word_embedding_file:
    :param tag_embedding_file:
    :param word_voc_file:
    :param tag_voc_file:
    :param regen:
    :return:
    """
    # If using regen model and need not re-generate embedding files,directly jump out without any operation
    if regen and ('_em' not in regen):
        return

    if not os.path.exists(embedding_root):
        os.mkdir(embedding_root)
    # 初始化word embedding
    if regen and regen != 'word_em':
        pass
    else:
        _init_word_embedding(word_embedding_file, word2vec_file, word_voc_file, overwrite=True)
    # 初始化tag embedding
    if regen and regen != 'tag_em':
        pass
    else:
        _init_tag_embedding(tag_embedding_file, tag_voc_file, overwrite=True)
