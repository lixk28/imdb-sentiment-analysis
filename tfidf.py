from operator import index
from dataset import *
import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

def get_tf(X, word_index):
  TF = {}
  for review_index in tqdm(range(len(X)), desc="Calculating tf", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
    TF[review_index] = {}
    review = X[review_index]
    for index_of_word in review:
      if index_of_word != 0:
        review = list(review)
        TF[review_index][index_of_word] = review.count(index_of_word) / len(review)
  return TF

def get_idf(X, word_index):
  IDF = {}
  if not os.path.exists("./models/idf.txt"):
    for index_of_word in tqdm(word_index.values(), desc="Calculating idf", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
      IDF[index_of_word] = 0
      for review in X:
        if index_of_word in review:
          IDF[index_of_word] += 1
    for index_of_word in IDF.keys():
      IDF[index_of_word] = np.log10(len(X) / (1 + IDF[index_of_word]))
    idf_file = open("./models/idf.txt", 'w')
    for index_of_word, idf in IDF.items():
      idf_file.write("{}\t{}\n".format(index_of_word, idf))
    idf_file.close()
  else:
    idf_file = open("./models/idf.txt", 'r')
    for line in idf_file:
      line = line.strip().split()
      index_of_word = int(line[0])
      idf = float(line[1])
      IDF[index_of_word] = idf
    idf_file.close()
  return IDF

def get_tfidf(TF, IDF):
  TF_IDF = {}
  for review_index in tqdm(TF.keys(), desc="Calculating tf-idf", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
    TF_IDF[review_index] = {}
    for index_of_word in TF[review_index].keys():
      TF_IDF[review_index][index_of_word] = TF[review_index][index_of_word] * IDF[index_of_word]
  return TF_IDF

def get_tf_embedding(X, word_index):
  TF = get_tf(X, word_index)
  tf_embedding_matrix = np.zeros((len(TF), len(word_index)+1))
  for review_index in TF.keys():
    for index_of_word in TF[review_index].keys():
     tf_embedding_matrix[review_index][index_of_word] = TF[review_index][index_of_word]
  normalize(X=tf_embedding_matrix, norm='l2', copy=False, return_norm=False)
  return tf_embedding_matrix

def get_tfidf_embedding(X, word_index):
  TF = get_tf(X, word_index)
  IDF = get_idf(X, word_index)
  TF_IDF = get_tfidf(TF, IDF)
  tfidf_embedding_matrix = np.zeros((len(TF_IDF), len(word_index)+1))
  # print(tfidf_embedding_matrix.shape)
  # for index_of_word in tqdm(word_index.values(), desc="Setting tf-idf embedding matrix", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
  #   for review_index in range(len(TF_IDF)):
  #     try:
  #       tfidf_embedding_matrix[index_of_word][review_index] = TF_IDF[review_index].get(index_of_word)
  #     except:
  #       continue

  for review_index in tqdm(TF_IDF.keys(), desc="Setting tf-idf embedding matrix", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
    for index_of_word in TF_IDF[review_index].keys():
      tfidf_embedding_matrix[review_index][index_of_word] = TF_IDF[review_index][index_of_word]
  # print(tfidf_embedding_matrix.shape)
  # lsa = TruncatedSVD(200)
  # tfidf_embedding_matrix = lsa.fit_transform(tfidf_embedding_matrix)
  normalize(X=tfidf_embedding_matrix, norm='l2', copy=False, return_norm=False)
  # print(tfidf_embedding_matrix.shape)
  return tfidf_embedding_matrix
