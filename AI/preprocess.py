import argparse
import sys

from config import *
from utils import build_text_label_file, segment_and_pos, split_train_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Not to add argument of labels file,always use that of input
    parser.add_argument("-a", "--account", type=str, help="Providing account to build text-labels files.")

    args = parser.parse_args()

    if args.account is None:
        print("Missing account,it must be provided.")
        sys.exit(1)
    else:
        # Todo:Check if account is valid,should create a collection to store all accounts?
        pass

    # Generate raw data:labels and text
    raw_data_file = RAW_DATA_FILE.format(name=args.account)
    build_text_label_file(args.account, raw_data_file)

    # Segment and part-of-speech
    corpus_file = raw_data_file
    seg_file = INPUT_SEGMENT_FILE.format(name=args.account)
    segment_and_pos(corpus_file, seg_file)

    # Split data into train and test set
    source_file = seg_file
    train_file = TRAIN_SEGMENT_FILE.format(name=args.account)
    test_file = TEST_SEGMENT_FILE.format(name=args.account)
    split_train_test(source_file, train_file, test_file, test_size=0.2, random_state=0)
