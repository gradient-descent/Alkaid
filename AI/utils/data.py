import os
import pickle
from collections import defaultdict

import numpy as np


def create_dictionary(items, dic_path, start=0, sort=True,
                      min_count=None, lower=False, overwrite=False):
    """
    Create dictionary and write to pkl
    :param items: list, [item_1, item_2, ...]
    :param dic_path:
    :param start: start index of vocabulary,default is 0
    :param sort:bool,if True,sort by frequency,else sort by items
    :param min_count:
    :param lower:bool
    :param overwrite:bool
    :return:
    """
    assert not dic_path.endswith('pk')
    if os.path.exists(dic_path) and not overwrite:
        return
    voc = dict()
    if sort:
        # Create dictionary
        dic = defaultdict(int)
        for item in items:
            item = item if (not lower) else item.lower()
            dic[item] += 1
        # Sort
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            index = i + start
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            voc[key] = index
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            index = i + start
            voc[item] = index

    file = open(dic_path, 'wb')
    pickle.dump(voc, file)
    file.close()


def map_item2id(items, voc, max_len, none_word=0, lower=False):
    """
    Map word/pos into id
    :param items: list, list to be mapped
    :param voc:
    :param max_len: int
    :param none_word: 未登录词标号,default 0
    :param lower:
    :return: arr: np.array, dtype=int32, shape=[max_len,]
    """
    assert type(none_word) == int
    arr = np.zeros((max_len,), dtype='int32')
    min_range = min(max_len, len(items))
    for i in range(min_range):  # 若items长度大于max_len，则被截断
        item = items[i] if not lower else items[i].lower()
        arr[i] = voc[item] if item in voc else none_word
    return arr
