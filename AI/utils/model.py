import pickle

import numpy as np

from .config import *
from .data import map_item2id
from .io import read_lines


def get_sentence_array(words_tags, word_voc, tag_voc):
    """

    :param words_tags:
    :param word_voc:
    :param tag_voc:
    :return: sentence_arr: np.array, list of word-id
    :return: tag_arr: np.array, list of pos-tags
    """
    words, pos_tags = [], []
    for item in words_tags:
        try:
            r_index = item.rindex('/')
            words.append(item[:r_index])
            pos_tags.append(item[r_index + 1:])
        except ValueError:
            # TODO:Write to log
            print('One line occurring \'{item}\' get \'ValueError: substring not found\''
                  .format(item=item))
    # sentence arr
    sentence_arr = map_item2id(
        words, word_voc, MAX_LEN, lower=True)
    # pos tags arr
    pos_tag_arr = map_item2id(
        pos_tags, tag_voc, MAX_LEN, lower=False)
    return sentence_arr, pos_tag_arr, len(words)


def load_embedding(word_embedding_file, tag_embedding_file):
    # Load word embedding
    with open(word_embedding_file, 'rb') as file_r:
        word_weights = pickle.load(file_r)
    # Load tag embedding
    with open(tag_embedding_file, 'rb') as file_r:
        tag_weights = pickle.load(file_r)
    return word_weights, tag_weights


def load_vocabulary(word_voc_file, tag_voc_file, labels_voc_file):
    with open(word_voc_file, 'rb') as file_r:
        word_voc = pickle.load(file_r)
    with open(tag_voc_file, 'rb') as file_r:
        tag_voc = pickle.load(file_r)
    with open(labels_voc_file, 'rb') as file_r:
        labels_voc = pickle.load(file_r)
    return word_voc, tag_voc, labels_voc


def init_data(lines, word_voc, tag_voc, labels_voc):
    data_count = len(lines)
    sentences = np.zeros((data_count, MAX_LEN), dtype='int32')
    tags = np.zeros((data_count, MAX_LEN), dtype='int32')
    sentence_actual_lengths = np.zeros((data_count,), dtype='int32')
    labels = np.zeros((data_count,), dtype='int32')
    instance_index = 0

    for i in range(data_count):
        label_text = lines[i].split(SEPARATOR)
        if len(label_text) < 2:
            label_text.append('')
        label = label_text[0]
        sentence = label_text[1]
        words_tags = sentence.split(' ')
        sentence_arr, tag_arr, actual_length = get_sentence_array(words_tags, word_voc, tag_voc)

        sentences[instance_index, :] = sentence_arr
        tags[instance_index, :] = tag_arr
        sentence_actual_lengths[instance_index] = actual_length
        labels[instance_index] = labels_voc[label] if label in labels_voc else 0
        instance_index += 1

    return sentences, tags, labels


def load_train_data(train_segment_file, word_voc, tag_voc, label_voc):
    return init_data(read_lines(train_segment_file), word_voc, tag_voc, label_voc)


def load_test_data(test_segment_file, word_voc, tag_voc, label_voc):
    sentences, tags, _ = init_data(read_lines(test_segment_file), word_voc, tag_voc, label_voc)
    return sentences, tags
