'''
  Load IMDB dataset of 50K Movie Reviews.
  Clean HTML tags.
  Unify word capitalization, remove stop words, low-frequency words, numbers, and punctuation.
'''
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stopwords_file = "./assets/stopwords-en.txt"
imdb_dataset = "./assets/IMDB Dataset.csv"

POSITIVE = 1
NEGATIVE = 0

def strip_html_tags(review: str) -> str:
  return BeautifulSoup(review, "lxml").get_text()

def remove_special_chars(review: str) -> str:
  # replace all the special chars and digits in the review
  return re.sub(r"[^A-Za-z]+" , ' ', review)

def tokenize(review: str) -> str:
  # just tokenize by white space
  token_list = review.strip().split()
  return ' '.join(token_list)

def remove_stopwords(review: str) -> str:
  # load stopwords
  stopwords_list = []
  file = open(stopwords_file, 'r')
  for line in file:
    line = line.strip()
    stopwords_list.append(line)
  file.close()
  # remove stopwords from review
  review_filtered = []
  for word in review.split():
    if word not in stopwords_list:
      review_filtered.append(word)
  # join words by space
  return ' '.join(review_filtered)

def preprocess(review):
  review = strip_html_tags(review) # strip html tags
  review = remove_special_chars(review) # remove all the special chars including digits and punctuation, leaving letters only
  review = review.lower() # to lower case
  review = tokenize(review) # tokenize
  review = remove_stopwords(review) # remove stopwords
  return review

def clean_dataset():
  df = pd.read_csv(imdb_dataset)
  print(df)
  for i in tqdm(range(len(df.index)), desc="Loading and preprocessing raw dataset", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
    df['review'].iloc[i] = preprocess(df['review'].iloc[i])
    df['sentiment'].iloc[i] = POSITIVE if df['sentiment'].iloc[i] == 'positive' else NEGATIVE
  df.columns = ['review', 'label']
  df.to_csv("./assets/processed_dataset.csv", index=False)
  return df

def get_word_index(X):
  '''
    Establish word-freq dict wordbag, remove low-frequency (< 50) words
    Then, build the mapping from each word to its unique index
  '''
  word_bag = {}  # mapping from word to its total frequency in all the reviews
  for review in X:
    for word in review:
      if word not in word_bag.keys():
        word_bag[word] = 0
      else:
        word_bag[word] += 1

  word_bag_sorted = sorted(word_bag.items(), key=lambda x: x[1], reverse=True)
  # print(word_bag_sorted[:100])

  word_index = {} # mapping from word to its index
  index = 1
  for word, freq in word_bag_sorted:
    if freq < 50:
      break
    word_index[word] = index
    index += 1

  return word_index

def load_dataset(sequence_length):
  '''
    X is reviews list, X[i] is the token list of the i-th review
    Y is the corresponding labels list of reviews, Y[i] is the label of review X[i]
  '''
  # read X, Y from processed data set, convert them to numpy ndarray
  dataset = pd.read_csv("./assets/processed_dataset.csv")
  X = dataset['review'].iloc[:].to_numpy()
  Y = dataset['label'].iloc[:].to_numpy()

  for i in range(len(X)):
    X[i] = X[i].split()

  word_index = get_word_index(X)

  for i in range(len(X)):
    for j in range(len(X[i])):
      if X[i][j] in word_index.keys():
        X[i][j] = word_index[X[i][j]]
      else:
        X[i][j] = 0
  
  X = pad_sequences(X, maxlen=sequence_length, dtype=np.int32, value=0, padding='post', truncating='post')

  return X, Y, word_index
