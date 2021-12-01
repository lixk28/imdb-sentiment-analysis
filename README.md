Midterm project for DCS301: Natural Language Processing taught by Prof. Quan (Xiaojun Quan) at Sun Yat-sen University.

Binary sentiment classification based on IMDB Dataset of 50k movie reviews.

Three different features used to feed neural network:
- pre-trained google news word2vec (with 1d cnn)
- tf (with a simple bpnn)
- tf-idf (with the same bpnn above)

Dataset split:
- 0 ~ 29999: train set
- 30000 ~ 39999: validation set
- 40000 ~ 49999: test set

At last, the three classification methods get about 87% ~ 88% accuracy with 0.3 loss approximately.

Usage:
```
usage: python3 main.py [-h] -f {w2v,tf,tf-idf}

optional arguments:
  -h, --help                                  show this help message and exit
  -f {w2v,tf,tf-idf}, --feed {w2v,tf,tf-idf}  use word2vec, tf or tf-idf to feed neural network
```

To run the program, you have to download `GoogleNews-vectors-negative300.bin.gz` from https://code.google.com/archive/p/word2vec/ and put it into `models` directory.
