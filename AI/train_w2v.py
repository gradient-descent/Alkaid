import argparse
import sys

from config import *
from word2vec import train_word2vec, bin2pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Not to add argument of labels file,always use that of input
    parser.add_argument("-a", "--account", type=str, help="Providing account to train word2vec model.")

    args = parser.parse_args()

    if args.account is None:
        print("Missing account,it must be provided.")
        sys.exit(1)
    else:
        # Todo:Check if account is valid,should create a collection to store all accounts?
        pass

    input_segment_file = INPUT_SEGMENT_FILE.format(name=args.account)
    sentences_file = SENTENCES_FILE.format(name=args.account)
    output_model = WORD2VEC_MODEL_FILE.format(name=args.account)
    output_word_vector = WORD2VEC_VECTOR_FILE.format(name=args.account)
    train_word2vec(
        input_segment_file,
        sentences_file,
        output_model,
        output_word_vector
    )

    word_vector_file = output_word_vector
    word2vec_file = WORD2VEC_FILE.format(name=args.account)
    bin2pkl(word_vector_file, word2vec_file)
