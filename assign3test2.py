# -*- coding: utf-8 -*-

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
import csv
import numpy as np
from string import maketrans
import string
import time

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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy
from keras.optimizers import SGD
from keras.layers import LSTM
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

Xinput = np.array(X)
Yinput = np.array(Y)

# # create model
model = Sequential()
# model.add(Dense(14, input_dim=14, init='glorot_uniform', activation='relu'))
# # model.add(Dropout(0.01))
# model.add(Dense(7, init='uniform', activation='relu'))
# # model.add(Dropout(0.01))
# model.add(Dense(1, init='uniform', activation='sigmoid'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#
# # Compile model
# optimizer = SGD(lr=0.1, momentum=0.8,nesterov=True)
# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#
# # Fit the model
# model.fit(Xinput, Yinput, epochs=600, batch_size=10, verbose=2)


# model.add(LSTM(150,input_shape=(14,1),return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(200,return_sequences=True))
# model.add(Dropout(0.2))
#
# model.add(LSTM(150,return_sequences=True))
# model.add(Dropout(0.2))
#
# model.add(Dense(28000, init='uniform', activation='sigmoid'))

model.add(Dense(14, input_dim=14,init='glorot_uniform', activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(14, init='glorot_uniform',activation='relu'))
model.add(Dense(14, init='glorot_uniform',activation='relu'))
model.add(Dense(14, init='glorot_uniform',activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(Xinput,Yinput,
          epochs=2000,
          batch_size=10,verbose=2)

# n_hours = 1
# n_features=14
# model.fit(np.reshape(Xinput, n_hours,n_features),Yinput,batch_size=100,nb_epoch=10,verbose=1,validation_split=0.05)
# model.fit(Xinput,Yinput,batch_size=100,nb_epoch=10,verbose=1,validation_split=0.05)


# 4. 评估模型
loss, accuracy = model.evaluate(np.array(X), np.array(Y), batch_size=10)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
# 5. 数据预测
probabilities = model.predict(np.array(X))
predictions111 = [float(round(x)) for x in probabilities]
accuracy = numpy.mean(predictions111 == Y)
print("Prediction Accuracy: %.2f%%" % (accuracy * 100))

# ----------------begin test predictions
with open('test.csv', 'rb') as testcsvFile:
    lines = csv.reader(testcsvFile)
    for line in lines:
        testlist.append(list(line))

    firstrow = testlist[0]
    del testlist[0]  # delete first row of train.csv
    testlistACGT = testlist
    # trainlistrow = [x[1] for x in trainlist]
    translatetable = maketrans('ACGT', '1234')
    print testlist
    for eachrow in testlist:
        eachrow[0] = string.atoi(eachrow[0])
        # print 'eachrow=' + eachrow[1]
        eachrow[1] = eachrow[1].translate(translatetable)
        # print 'new_eachrow=' + eachrow[1]
        Xstringtest = '-'.join(eachrow[1]).split('-')
        Xstringtest = map(eval, Xstringtest)
        # print Xstring
        XForTest.append(Xstringtest)
        # eachrow[2] = string.atoi(eachrow[2])
        # Yfortest.append(eachrow[2])
    print testlist
    print XForTest
    # calculate predictions
    predictions = model.predict(np.array(XForTest))
    # round predictions
    rounded = [int(round(x[0])) for x in predictions]
    print(rounded)
    # ------------------output test predictions to csv
    idofdna = [x[0] for x in testlistACGT]
    outputlist = np.c_[np.array(idofdna), np.array(rounded)]
    firstrowtest = ['id', 'prediction']
    # numpy.savetxt('outputtest.csv', outputlist, delimiter = ',',fmt="%f,%f,%f")
    with open("outputtest22222.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(firstrowtest)
        writer.writerows(outputlist)