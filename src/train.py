#!/usr/bin/env python
#coding:utf-8
import os
import numpy as np

import keras
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from network_model import DNNModel
from loss_history import LossHistory

REPOSITORY_PATH = str("/".join(os.path.abspath(__file__).split("/")[0:-2]))
TRAIN_DATA_DIR = REPOSITORY_PATH + "/dataset/lab_members/"
WEIGHT_FILE_PATH = REPOSITORY_PATH + "/model/lab_members.hdf5"
LABEL_INFO = []

def create_train_data():
	files = os.listdir(TRAIN_DATA_DIR)
	label_num = len(files)
	landmarks = []
	labels = []

	for i, file in enumerate(files):
		with open(TRAIN_DATA_DIR+file, "r") as f:
			lines = f.read().split("\n")
		f.close()

		for line in lines:
			marks= line.split(" ")
			landmarks.append(np.array(marks).flatten())
			labels.append(i)

		LABEL_INFO.append(file.split(".csv")[0])

	return landmarks, labels, label_num


def main():
	landmarks, labels, label_num = create_train_data()

	landmarks = np.asarray(landmarks).astype("float32")
	labels = np_utils.to_categorical(labels, label_num)

	model = DNNModel(label_num).model
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

	#学習checkpointの設定（periodごとに精度を評価し、よければweightを保存）
	check_point = ModelCheckpoint(
	        filepath=WEIGHT_FILE_PATH,
	        monitor="val_loss",
	        save_best_only=True,
	        period=1,
	    )

	history = model.fit(landmarks, labels,
	    batch_size=64,
	    epochs=3000,
		validation_split=0.1,
		callbacks=[check_point, LossHistory()])

	model.save_weights(WEIGHT_FILE_PATH)
	print "model was saved."


	print LABEL_INFO



def old():

	carlos_data_path = REPOSITORY_PATH + "/dataset/carlos.csv"
	rowan_data_path = REPOSITORY_PATH  + "/dataset/rowan.csv"

	weight_file_path = REPOSITORY_PATH + "/model/weight_sigmoid.hdf5"

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


if __name__ == "__main__":
	main()

