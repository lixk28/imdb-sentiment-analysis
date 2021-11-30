from dataset import *
from sklearn.preprocessing import normalize
from gensim.models import KeyedVectors

import numpy as np
# import time
# import logging

def get_w2v_embedding(word_index):
  w2v_model = KeyedVectors.load_word2vec_format("./models/GoogleNews-vectors-negative300.bin.gz", binary=True, limit=2000000)
  w2v_embedding_matrix = np.zeros((len(word_index) + 1, w2v_model.vector_size))
  for word, index in word_index.items():
    try:
      w2v_embedding_matrix[index] = w2v_model[word]
    except:
      continue  # if word is not in w2v_model, then it's embedding is set to zero
  normalize(X=w2v_embedding_matrix, norm='l2', copy=False, return_norm=False)
  return w2v_embedding_matrix