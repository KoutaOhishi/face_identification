#!/usr/bin/env python
#coding:utf-8
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Dropout


class DNNModel():
    def __init__(self, label_num):
        self.model = Sequential()
        #self.model.add(Dense(1024, input_shape=(136,)))
        self.model.add(Dense(1024, input_dim=136))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))

        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))

        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))

        self.model.add(Dense(128))
        self.model.add(Activation('relu'))

        self.model.add(Dense(label_num))#正解ラベルの数に合わせる
        self.model.add(Activation('softmax'))
        #self.model.add(Activation('sigmoid'))

