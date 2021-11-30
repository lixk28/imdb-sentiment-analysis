import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Embedding, Conv2D, MaxPool1D, MaxPool2D, Dense
from tensorflow.keras.layers import Flatten, Dropout, Concatenate, Reshape
import numpy as np

def cnn(embedding_matrix, embedding_dim, sequence_length, vocab_size):
  model = Sequential()
  model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length, \
                      weights=[embedding_matrix], trainable=False))

  model.add(Conv1D(embedding_dim, 3, activation='relu', padding='same'))
  model.add(MaxPool1D(2))
  model.add(Dropout(0.2))

  model.add(Conv1D(embedding_dim // 2, 3, activation='relu', padding='same'))
  model.add(MaxPool1D(2))
  model.add(Dropout(0.2))

  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
  model.summary()
  return model

def nn(vocab_size):
  model = models.Sequential()
  model.add(Dense(256, activation='relu', input_shape=(vocab_size,)))
  model.add(Dropout(0.2))
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
  model.summary()
  return model
