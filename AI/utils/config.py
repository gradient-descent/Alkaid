import sys

sys.path.append('/home/shuo/Dropbox/projects/spiders/oa_spider')
from db import MongoManager
from config import EMBEDDING_SIZE, MAX_LEN as G_MAX_LEN

mongo_manager = MongoManager()

# --- word2vec/corpus/vocabulary param ---
W2V_DIM = EMBEDDING_SIZE
SEPARATOR = ' $$$$$$$$$$ '
WORD_VOC_START = 2
TAG_VOC_START = 1
TAG_DIM = 64

# --- training param ---
MAX_LEN = G_MAX_LEN

# --- rnn param ---
RNN_HIDDEN_SIZE = 64
