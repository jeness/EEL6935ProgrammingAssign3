#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
import csv
import numpy as np
from string import maketrans
trainlist=[]
DNAdict = {'A':1,'C':2,'G':3,'T':4}
with open('train.csv','rb') as traincsvFile:
    lines=csv.reader(traincsvFile)
    for line in lines:
        trainlist.append(list(line))

    firstrow = trainlist[0]
    del trainlist[0] #delete first row of train.csv
    print trainlist

    trainlistrow = [x[1] for x in trainlist]
    translatetable = maketrans('ACGT', '1234')
    print trainlist
    for eachrow in trainlist:
        print 'eachrow=' + eachrow[1]
        eachrow[1] = eachrow[1].translate(translatetable)
        print 'new_eachrow=' + eachrow[1]

    print trainlist
    # l.append(line)
    # l.remove(l[0])
    # print l

    # X= dataset[:,0:8]
