import multiprocessing
import pickle

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence

from .config import *


def word2vec_model(input_text, output_model, output_word_vector):
    """
    When the sample sizes are small,perhaps need to adjust 'min_count',
    otherwise should throw "you must first build vocabulary before training the model" error.
    To keep this vector's dimensions consistent with embedding-size of LSTM model,
    using EMBEDDING_SIZE in w2v_config.py .
    :param input_text:
    :param output_model:
    :param output_word_vector:
    :return:
    """
    model = Word2Vec(
        LineSentence(input_text),
        size=WORD2VEC_SIZE,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        workers=multiprocessing.cpu_count()
    )
    model.save(output_model, ignore=[])
    model.wv.save_word2vec_format(output_word_vector, binary=WORD2VEC_FORMAT_BINARY)


def bin2pkl(word_vector_file, word2vec_file):
    model = KeyedVectors.load_word2vec_format(word_vector_file, binary=WORD2VEC_FORMAT_BINARY)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    with open(word2vec_file, 'wb') as file_w:
        pickle.dump(word_dict, file_w)
        print(file_w.name)
