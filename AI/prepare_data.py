import argparse
import pickle
import sys
import time

from config import *
from utils import init_vocabulary, init_embedding


def demo(word_embedding_file):
    with open(word_embedding_file, 'rb') as file:
        temp = pickle.load(file)
    print(temp.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Not to add argument of labels file,always use that of input
    parser.add_argument("-a", "--account", type=str, help="Provide an oa account.")
    # Selectively execute vocabulary/embedding initialization
    parser.add_argument("-r", "--regen", type=str,
                        choices=['word_voc', 'tag_voc', 'label_voc', 'word_em', 'tag_em'],
                        help="Select one of vocabulary or embedding of word/tag/label to re-generate.")

    args = parser.parse_args()

    if args.account is None:
        print("Missing account,it must be provided.")
        sys.exit(1)
    else:
        # Todo:Check if account is valid,should create a collection to store all accounts?
        pass

    t0 = time.time()

    input_segment_file = INPUT_SEGMENT_FILE.format(name=args.account)
    word_voc_file = WORD_VOC_FILE.format(name=args.account)
    tag_voc_file = TAG_VOC_FILE.format(name=args.account)
    label_voc_file = LABEL_VOC_FILE.format(name=args.account)
    init_vocabulary(
        args.account,
        input_segment_file,
        word_voc_file,
        tag_voc_file,
        label_voc_file,
        args.regen
    )

    embedding_root = EMBEDDING_PATH
    word2vec_file = WORD2VEC_FILE.format(name=args.account)
    word_embedding_file = WORD_EMBEDDING_FILE.format(name=args.account)
    tag_embedding_file = TAG_EMBEDDING_FILE.format(name=args.account)
    init_embedding(
        embedding_root,
        word2vec_file,
        word_embedding_file,
        tag_embedding_file,
        word_voc_file,
        tag_voc_file,
        args.regen
    )

    demo(word_embedding_file)

    print('Done in %.1fs!' % (time.time() - t0))
