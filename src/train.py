#!/usr/bin/env python
#coding:utf-8
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import os
#import seaborn as sns; sns.set()

#from tqdm import tqdm
from network_model import DNNModel
#from io import open
import keras
#from keras import backend
#from keras.models import Sequential
#from keras.layers import Activation, Dense
#from keras.layers.recurrent import LSTM
#from keras.layers import Dropout
#from keras.preprocessing import sequence
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils import to_categorical
from keras.utils import np_utils
#from keras import metrics
#import math
#import datetime

carlos_data_path = "/home/macbook-air/face_identification/dataset/carlos.csv"
rowan_data_path = "/home/macbook-air/face_identification/dataset/rowan.csv"

weight_file_path = "/home/macbook-air/face_identification/model/weight.hdf5"

landmarks = []
labels = []

with open(carlos_data_path, "r") as f:
	carlos_lines = f.read().split("\n")	
	f.close()

with open(rowan_data_path, "r") as f:
	rowan_lines = f.read().split("\n")
	f.close()

for i in range(len(carlos_lines)-1):
	carlos_line = carlos_lines[i].split(" ")
	landmarks.append(np.array(carlos_line).flatten())
	labels.append(0)

for i in range(len(rowan_lines)-1):
	rowan_line = rowan_lines[i].split(" ")
	landmarks.append(np.array(rowan_line).flatten())
	labels.append(1)

#for i in range(len(labels)):
	#print landmarks[i]
	#print labels[i]
	#print "---"


#print np.array(landmarks).shape


landmarks = np.asarray(landmarks).astype("float32")
labels = np_utils.to_categorical(labels, 2)

model = DNNModel().model
model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

history = model.fit(landmarks, labels,
    batch_size=64,
    epochs=3000)

model.save_weights(weight_file_path)
print "model was saved."


