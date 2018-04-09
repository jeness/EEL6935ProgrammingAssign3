# -*- coding: utf-8 -*-

import csv
import numpy as np
from string import maketrans
import string
from keras.utils import plot_model
import pydot
import graphviz
trainlist=[]
testlist=[]
X = []
Y = []
XForTest = []
Yfortest = []
#DNAdict = {'A':1,'C':2,'G':3,'T':4}
with open('train.csv','rb') as traincsvFile:
    lines=csv.reader(traincsvFile)
    for line in lines:
        trainlist.append(list(line))

    firstrow = trainlist[0]
    del trainlist[0] #delete first row of train.csv

    #trainlistrow = [x[1] for x in trainlist]
    translatetable = maketrans('ACGT', '1234')
    print trainlist
    for eachrow in trainlist:
        eachrow[0] = string.atoi(eachrow[0])
        # print 'eachrow=' + eachrow[1]
        eachrow[1] = eachrow[1].translate(translatetable)
        #print 'new_eachrow=' + eachrow[1]
        Xstring = '-'.join(eachrow[1]).split('-')
        Xstring = map(eval, Xstring)
        #print Xstring
        X.append(Xstring)
        eachrow[2] = string.atoi(eachrow[2])
        Y.append(eachrow[2])
    print trainlist
    print Y
    print X
#-----------------------------------begin train model
    from keras.models import Sequential
    import numpy
    from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten, Embedding, GRU
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # create model
    model = Sequential()

    model.add(Dense(14, input_dim=14, init='glorot_uniform', activation='relu'))
    # model.add(Dropout(0.01))
    model.add(Dense(14, init='glorot_uniform', activation='relu'))
    # model.add(Dropout(0.01))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(np.array(X), np.array(Y), epochs=700, batch_size=10, verbose=2)
    # calculate predictions
    # predictions = model.predict(np.array(X))
    # # round predictions
    # rounded = [round(x[0]) for x in predictions]
    # print(rounded)
    # plot_model(model, to_file='model.png')

    # 4. evaluate
    loss, accuracy = model.evaluate(np.array(X), np.array(Y))
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
    # 5. predict
    probabilities = model.predict(np.array(X))
    predictions111 = [float(round(x)) for x in probabilities]
    accuracy = numpy.mean(predictions111 == Y)
    print("Prediction Accuracy: %.2f%%" % (accuracy * 100))
    #----------------begin test predictions
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
        outputlist = np.c_[np.array(idofdna),np.array(rounded)]
        firstrowtest = ['id', 'prediction']
        # numpy.savetxt('outputtest.csv', outputlist, delimiter = ',',fmt="%f,%f,%f")
        with open("outputtest.csv", "wb") as f:
            writer = csv.writer(f)
            writer.writerow(firstrowtest)
            writer.writerows(outputlist)