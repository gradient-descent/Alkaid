import os

# Public parameters
RESOURCES_PATH = os.path.dirname(os.path.realpath(__file__)) + '/' + 'resources'
DATA_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/' + 'data'
RAW_PATH = DATA_ROOT + '/' + 'raw'
CORPUS_PATH = DATA_ROOT + '/' + 'corpus'
VOCABULARY_PATH = DATA_ROOT + '/' + 'vocabulary'
EMBEDDING_PATH = DATA_ROOT + '/' + 'embedding'
RESULTS_ROOT = DATA_ROOT + '/' + 'results'
STOP_WORDS_FILE = RESOURCES_PATH + '/' + 'stop_words.txt'

# Word2vec parameters
USE_PRE_TRAINED_W2V = True

# Raw data:input and labels
RAW_DATA_FILE = RAW_PATH + '/' + '{name}.raw'

# Corpus:segment and part-of-speech RAW_DATA_FILE
INPUT_SEGMENT_FILE = CORPUS_PATH + '/' + '{name}.seg.input'
TRAIN_SEGMENT_FILE = CORPUS_PATH + '/' + '{name}.train.seg.input'
TEST_SEGMENT_FILE = CORPUS_PATH + '/' + '{name}.test.seg.input'

# Vocabulary
WORD_VOC_FILE = VOCABULARY_PATH + '/' + '{name}.title.word_voc.pkl'
TAG_VOC_FILE = VOCABULARY_PATH + '/' + '{name}.title.tag_voc.pkl'
LABEL_VOC_FILE = VOCABULARY_PATH + '/' + '{name}.title.label_voc.pkl'

# Embedding
SENTENCES_FILE = EMBEDDING_PATH + '/' + '{name}.sentences.txt'
WORD2VEC_MODEL_FILE = EMBEDDING_PATH + '/' + '{name}.title.model'
WORD2VEC_VECTOR_FILE = EMBEDDING_PATH + '/' + '{name}.title.vector.bin'
WORD2VEC_FILE = EMBEDDING_PATH + '/' + '{name}.title.word2vec.pkl'
WORD_EMBEDDING_FILE = EMBEDDING_PATH + '/' + '{name}.title.word.embedding.pkl'
TAG_EMBEDDING_FILE = EMBEDDING_PATH + '/' + '{name}.title.tag.embedding.pkl'

# Global train/test data set parameters
MAX_LEN = 300
BATCH_SIZE = 64
NB_EPOCH = 30
KEEP_PROB = 0.5
WORD_KEEP_PROB = 0.9
TAG_KEEP_PROB = 0.9
MAX_GRAD_NORM = 5
KFOLD_N_SPLIT = 10

# Global neural network parameters
NB_LABELS = 4
EMBEDDING_SIZE = 256
