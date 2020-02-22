#!/usr/bin/env python
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import keras


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        # コンストラクタに保持用の配列を宣言しておく
        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        # 配列にEpochが終わるたびにAppendしていく
        self.train_acc.append(logs['accuracy'])
        self.val_acc.append(logs['val_accuracy'])
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])

        # グラフ描画部
        #plt.figure(num=1, clear=True)
        plt.figure(num=1)
        plt.clf() #図をクリア
        plt.title('Learning Curve')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(self.train_acc, label='train', color=(0,0,1))
        plt.plot(self.val_acc, label='validation', color=(0,1,0))
        plt.legend()#判例を表示
        plt.pause(0.01)
