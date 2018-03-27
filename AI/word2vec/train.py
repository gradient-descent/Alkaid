import logging
import sys

from config import *
from .model import word2vec_model
from .w2v_utils import extract_sentences


def train_word2vec(
        input_segment_file,
        sentences_file,
        output_model,
        output_word_vector,
        use_log=True):
    """

    :param input_segment_file:
    :param sentences_file:
    :param output_model:
    :param output_word_vector:
    :param use_log:
    :return:
    """
    extract_sentences(input_segment_file, sentences_file)
    if use_log:
        program = os.path.basename(sys.argv[0])
        logger = logging.getLogger(program)

        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(level=logging.INFO)
        logger.info("running %s" % ' '.join(sys.argv))

    word2vec_model(sentences_file, output_model, output_word_vector)
