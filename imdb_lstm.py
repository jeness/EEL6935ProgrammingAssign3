from __future__ import print_function

import csv
import numpy as np
from string import maketrans
import string

trainlist = []
testlist = []
testlistACGT = []
X = []
Y = []
XForTest = []
Yfortest = []
# DNAdict = {'A':1,'C':2,'G':3,'T':4}
with open('train.csv', 'rb') as traincsvFile:
    lines = csv.reader(traincsvFile)
    for line in lines:
        trainlist.append(list(line))

    firstrow = trainlist[0]
    del trainlist[0]  # delete first row of train.csv

    # print trainlist
    # trainlistrow = [x[1] for x in trainlist]
    translatetable = maketrans('ACGT', '1234')
    # print trainlist
    for eachrow in trainlist:
        eachrow[0] = string.atoi(eachrow[0])
        # print 'eachrow=' + eachrow[1]
        eachrow[1] = eachrow[1].translate(translatetable)
        # print 'new_eachrow=' + eachrow[1]
        Xstring = '-'.join(eachrow[1]).split('-')
        Xstring = map(eval, Xstring)
        # print Xstring
        X.append(Xstring)
        eachrow[2] = string.atoi(eachrow[2])
        Y.append(eachrow[2])
    # print trainlist
    # print Y
    # print X


'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

Xinput = np.array(X)
Yinput = np.array(Y)
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(Xinput, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, input_dim=14,activation='relu'))
# model.add(Dense(1, input_dim=14,activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(Xinput, Yinput,
          batch_size=batch_size,
          epochs=15,
          validation_data=(Xinput, Xinput), verbose=2)
score, acc = model.evaluate(Xinput, Yinput,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)