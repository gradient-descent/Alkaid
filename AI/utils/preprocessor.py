from jieba import posseg
from sklearn.model_selection import train_test_split

from .config import mongo_manager, SEPARATOR
from .io import read_lines

raw_train_collection = '{account_id}_raw_train_data'
labels_collection = '{account_id}_labels'


def build_text_label_file(account_id, text_file):
    """

    :param account_id:
    :param text_file:
    :return:
    """
    collection = raw_train_collection.format(account_id=account_id)
    all_text = []
    all_labels = []
    for r in mongo_manager.find(collection=collection):
        all_text.append(r['title'])
        all_labels.append(r['label'])

    if not all_text:
        print('This account may not be collected.')
        sys.exit(1)

    with open(text_file, 'w') as f:
        for idx in range(len(all_labels)):
            text = all_text[idx].strip()
            label = all_labels[idx].strip()
            f.write(label + SEPARATOR + text + '\n')

    print('All the text and labels dumped to files.\n')


def _segment_and_tag(words):
    """

    :param words:
    :return:
    """
    return list(posseg.cut(words))


def segment_and_pos(corpus_file, seg_file):
    """
    Segment and part-of-speech
    :param corpus_file:
    :param seg_file:
    :return:
    """
    lines = read_lines(corpus_file)
    index = 1
    with open(seg_file, 'w') as f:
        for line in lines:
            label_text = line.split(SEPARATOR)
            if len(label_text) < 2:
                print('Line %d\'s text is null.' % index)
                label_text.append('')
            word_tag = label_text[1]
            word_segment = _segment_and_tag(word_tag)
            f.write('%s' % str(label_text[0]))
            f.write('%s' % SEPARATOR)
            f.write('%s\n' % ' '.join([str(i) for i in word_segment]))
            index += 1

    print('All data have been segmented and tagged.\n')


def split_train_test(source_file, train_file, test_file, test_size=0.2, random_state=0):
    """

    :param source_file:
    :param train_file:
    :param test_file:
    :param test_size:
    :param random_state:
    :return:
    """
    lines = read_lines(source_file)
    X_train, X_test = train_test_split(lines, test_size=test_size, random_state=random_state)
    with open(train_file, 'w') as f_train:
        for train_item in X_train:
            f_train.write('%s\n' % str(train_item))
    with open(test_file, 'w') as f_test:
        for test_item in X_test:
            f_test.write('%s\n' % str(test_item))

    print('All data have been split into train and test set.\n')


def get_labels_vocabulary(account_id):
    collection = labels_collection.format(account_id=account_id)
    labels = []
    for r in mongo_manager.find(collection=collection):
        labels.append(r['name'])

    return labels
