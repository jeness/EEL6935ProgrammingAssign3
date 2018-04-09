import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np
import random
import csv
import copy

dati = []

#read data:
with open ('test.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		dati.append(row[1])
del dati[0]
input = np.zeros((400, 14, 4))

#convert letters into vectors:
def switch(letter = ''):
    if letter == 'A':
        return np.array([1, 0, 0, 0])
    elif letter == 'C':
        return np.array([0, 1, 0, 0])
    elif letter == 'G':
        return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])

#export to 'input':
for i in range(400):
    for j in range(14):
        vec = copy.copy(switch(dati[i][j]))
        input[i][j] = vec


model = load_model("./model.h5")
output = model.predict(input,batch_size = 100,verbose = 0,steps = None)
result = np.zeros((400,2), dtype = int)
for i in range(400):
	result[i][0] = int(i)
	result[i][1] = int(np.argmax(output[i]))


print(result)

writer = csv.writer(open('result.csv','w'))
title = ['id', 'prediction']
writer.writerow(title)
for row in result:
	writer.writerow(row)