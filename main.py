from dataset import *
from tfidf import *
from word2vec import *
from model import *
import os
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

sequence_length = 200

def history_plot(history, model_name):
  plt.title('Model accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.grid(linestyle='--')
  plt.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
  plt.savefig("./images/{}_accuracy.png".format(model_name))
  plt.show()
  plt.close()

  plt.title('Model loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.grid(linestyle='--')
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.savefig("./images/{}_loss.png".format(model_name))
  plt.show()
  plt.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--feed', type=str, required=True, \
        choices=['w2v', 'tf', 'tf-idf'], help='use word2vec, tf or tf-idf to feed textcnn')
  args = parser.parse_args()

  # clean dataset
  # remove html tags, special chars, and numbers
  if not os.path.exists("./assets/processed_dataset.csv"):  # if already cleaned
    clean_dataset()

  # load dataset from processed dataset
  # X is data, Y is label
  X, Y, word_index = load_dataset(sequence_length)

  # use word2vec or tf-idf to feed textcnn
  if args.feed == 'w2v':
    print("use word2vec to train")

    # split dataset
    # 0     - 29999: train set
    # 30000 - 39999: validation set
    # 40000 - 50000: test set
    X_train, Y_train = X[0:30000], Y[0:30000]
    X_valid, Y_valid = X[30000:40000], Y[30000:40000]
    X_test, Y_test   = X[40000:50000], Y[40000:50000]

    # get w2v embedding matrix
    w2v_embedding_matrix = get_w2v_embedding(word_index)

    # create text cnn model
    model = cnn(w2v_embedding_matrix, w2v_embedding_matrix.shape[1], sequence_length, len(word_index)+1)
    checkpoint = ModelCheckpoint('./checkpoint/w2v.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # compile and train model
    print("Traning Model...")
    history = model.fit(X_train, Y_train, batch_size=30, epochs=4, verbose=1, callbacks=[checkpoint], validation_data=(X_valid, Y_valid))
  
    # evaluate model
    print("Evaluate on test data")
    results = model.evaluate(X_test, Y_test, batch_size=50)
    print("test loss, test acc:", results)

    # visualize training history
    history_plot(history, 'w2v')

  elif args.feed == 'tf-idf':
    print("use tf to train")
    tfidf_embedding_matrix = get_tfidf_embedding(X, word_index)

    X_train, Y_train = tfidf_embedding_matrix[0:30000], Y[0:30000]
    X_valid, Y_valid = tfidf_embedding_matrix[30000:40000], Y[30000:40000]
    X_test, Y_test = tfidf_embedding_matrix[40000:50000], Y[40000:50000]

    # create text cnn model
    model = nn(len(word_index)+1)
    checkpoint = ModelCheckpoint('./checkpoint/tf-idf.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # compile and train model
    print("Traning Model...")
    history = model.fit(X_train, Y_train, batch_size=64, epochs=4, verbose=1, callbacks=[checkpoint], validation_data=(X_valid, Y_valid))
  
    # evaluate model
    print("Evaluate on test data")
    results = model.evaluate(X_test, Y_test, batch_size=50)
    print("test loss, test acc:", results)

    # visualize training history
    history_plot(history, 'tf-idf')

  elif args.feed == 'tf':
    print("use tf to train")
    tf_embedding_matrix = get_tf_embedding(X, word_index)

    X_train, Y_train = tf_embedding_matrix[0:30000], Y[0:30000]
    X_valid, Y_valid = tf_embedding_matrix[30000:40000], Y[30000:40000]
    X_test, Y_test = tf_embedding_matrix[40000:50000], Y[40000:50000]

    # create text cnn model
    model = nn(len(word_index)+1)
    checkpoint = ModelCheckpoint('./checkpoint/tf.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # compile and train model
    print("Traning Model...")
    history = model.fit(X_train, Y_train, batch_size=64, epochs=4, verbose=1, callbacks=[checkpoint], validation_data=(X_valid, Y_valid))
  
    # evaluate model
    print("Evaluate on test data")
    results = model.evaluate(X_test, Y_test, batch_size=50)
    print("test loss, test acc:", results)

    # visualize training history
    history_plot(history, 'tf')

