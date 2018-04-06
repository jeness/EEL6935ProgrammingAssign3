#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
import csv
import numpy as np
from string import maketrans
import string
trainlist=[]
X = []
Y = []
DNAdict = {'A':1,'C':2,'G':3,'T':4}
with open('train.csv','rb') as traincsvFile:
    lines=csv.reader(traincsvFile)
    for line in lines:
        trainlist.append(list(line))

    firstrow = trainlist[0]
    del trainlist[0] #delete first row of train.csv

    trainlistrow = [x[1] for x in trainlist]
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
    from keras.layers import Dense
    import numpy

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)


    # X= dataset[:,0:8]
