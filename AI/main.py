import argparse
import codecs
import sys
from time import time

import numpy as np
from sklearn.model_selection import KFold

from config import *
from utils import load_embedding, load_test_data, load_train_data, load_vocabulary, read_lines


def predict(
        account,
        selected_model,
        word_embedding_file,
        tag_embedding_file,
        word_voc_file,
        tag_voc_file,
        label_voc_file,
        train_segment_file,
        test_segment_file):
    word_weights, tag_weights = load_embedding(word_embedding_file, tag_embedding_file)
    word_voc, tag_voc, label_voc = load_vocabulary(
        word_voc_file, tag_voc_file, label_voc_file)

    # train data
    sentences, tags, labels = load_train_data(
        train_segment_file, word_voc, tag_voc, label_voc)
    seed = 137
    np.random.seed(seed)
    np.random.shuffle(sentences)
    np.random.seed(seed)
    np.random.shuffle(tags)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # load data
    sentences_test, tags_test = load_test_data(
        test_segment_file, word_voc, tag_voc, label_voc)
    labels_test = None

    # clear result
    results_path = RESULTS_ROOT + '/' + account + '/' + selected_model + '/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    command = 'rm -rf {results_path}/*'.format(results_path=results_path)
    os.popen(command)

    kf = KFold(n_splits=KFOLD_N_SPLIT)
    train_indices, dev_indices = [], []
    for train_index, dev_index in kf.split(labels):
        train_indices.append(train_index)
        dev_indices.append(dev_index)
    for num in range(KFOLD_N_SPLIT):
        train_index, dev_index = train_indices[num], dev_indices[num]
        sentences_train, sentences_dev = sentences[train_index], sentences[dev_index]
        tags_train, tags_dev = tags[train_index], tags[dev_index]
        labels_train, labels_dev = labels[train_index], labels[dev_index]

        if selected_model == 'cnn':
            from models import CNNModel
            result_file = results_path + 'result.txt'
            epoch_results_path = results_path + 'epoch' + '/'
            if not os.path.exists(epoch_results_path):
                os.makedirs(epoch_results_path)
            model = CNNModel(
                MAX_LEN, word_weights, tag_weights,
                epoch_results_path=epoch_results_path,
                result_file=result_file,
                label_voc=label_voc
            )
            model.fit(
                sentences_train, tags_train, labels_train,
                sentences_dev, tags_dev, labels_dev,
                sentences_test, tags_test, labels_test,
                BATCH_SIZE, NB_EPOCH,
                keep_prob=KEEP_PROB,
                word_keep_prob=WORD_KEEP_PROB,
                tag_keep_prob=TAG_KEEP_PROB
            )

            print(model.get_best_score())
            [p_test, r_test, f_test], nb_epoch = model.get_best_score()

            best_results_path = results_path + 'best' + '/'
            if not os.path.exists(best_results_path):
                os.makedirs(best_results_path)
            command = \
                'cp' + ' ' + epoch_results_path + '{epoch}.csv'.format(
                    epoch=nb_epoch + 1) + ' ' + best_results_path + '{best}'.format(
                    best=num)
            print(command)
            os.popen(command)
            print(p_test, r_test, f_test, '\n')


def _init_result(results_path):
    best_results_path = results_path + 'best' + '/'
    if not os.path.exists(best_results_path):
        os.mkdir(best_results_path)
    labels = []
    for i in range(KFOLD_N_SPLIT):
        lines = read_lines(best_results_path + '{best}'.format(best=i))
        temp = []
        for line in lines:
            label = line.split(',')[1]
            temp.append(label)
        labels.append(temp)
    return labels


def merge(results_path):
    data_array = _init_result(results_path)
    data_count = len(data_array[0])
    label_type_count = NB_LABELS
    labels = np.zeros((data_count, label_type_count))
    for data in data_array:
        for i, label in enumerate(data):
            label_id = int(label) - 1
            labels[i][label_id] += 1
    # 取众数
    final_labels = []
    for item in labels:
        label = item.argmax() + 1
        final_labels.append(label)

    # clear result
    command = 'rm -rf {results_path}/*'.format(results_path=results_path)
    os.popen(command)

    grade_file = results_path + 'intgrade.csv'
    with codecs.open(grade_file, 'w', encoding='utf-8') as file_w:
        for i, label in enumerate(final_labels):
            file_w.write('%d,%d\n' % (i + 1, label))
        print('Result: %s' % file_w.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Not to add argument of labels file,always use that of input
    parser.add_argument("-a", "--account", type=str, help="Provide an account to train.")
    parser.add_argument("-m", "--model", type=str, choices=['cnn'], default='cnn',
                        help="Select a model to run.")

    args = parser.parse_args()

    if args.account is None:
        print("Missing account,it must be provided.")
        sys.exit(1)
    else:
        # Todo:Check if account is valid,should create a collection to store all accounts?
        pass

    t0 = time()

    _word_embedding_file = WORD_EMBEDDING_FILE.format(name=args.account)
    _tag_embedding_file = TAG_EMBEDDING_FILE.format(name=args.account)
    _word_voc_file = WORD_VOC_FILE.format(name=args.account)
    _tag_voc_file = TAG_VOC_FILE.format(name=args.account)
    _label_voc_file = LABEL_VOC_FILE.format(name=args.account)
    _train_segment_file = TRAIN_SEGMENT_FILE.format(name=args.account)
    _test_segment_file = TEST_SEGMENT_FILE.format(name=args.account)
    _results_path = RESULTS_ROOT + '/' + args.account + '/' + args.model + '/'
    if not os.path.exists(_results_path):
        os.mkdir(_results_path)

    # predict test data
    predict(
        args.account,
        args.model,
        _word_embedding_file,
        _tag_embedding_file,
        _word_voc_file,
        _tag_voc_file,
        _label_voc_file,
        _train_segment_file,
        _test_segment_file
    )

    # merge
    merge(_results_path)

    print('Done in %ds!' % (time() - t0))
