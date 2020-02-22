#!/usr/bin/env python
#coding:utf-8
import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
from network_model import DNNModel

REPOSITORY_PATH = str("/".join(os.path.abspath(__file__).split("/")[0:-2]))
CASCADE_PATH = REPOSITORY_PATH + "/model/haarcascade_frontalface_alt.xml"
MODEL_PATH = REPOSITORY_PATH + "/model/shape_predictor_68_face_landmarks.dat"
WEIGHT_FILE_PATH = REPOSITORY_PATH + "/model/lab_members.hdf5"

LABEL_INFO = [
	"ohishi",
	"tsuyuzaki",
	"tsurue"
]

def camera():
	cap = cv2.VideoCapture(0)
	cascade = cv2.CascadeClassifier(CASCADE_PATH)
	predictor = dlib.shape_predictor(MODEL_PATH)
	detector = dlib.get_frontal_face_detector()

	model = DNNModel(len(LABEL_INFO)).model
	model.load_weights(WEIGHT_FILE_PATH)
	graph = tf.get_default_graph()



	while True:
		_, img = cap.read()

		gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		faces = cascade.detectMultiScale(gray_img, minSize=(30, 30))

		if len(faces) != 0:
			for(x, y, width, height) in faces:
				cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 255), 1)
				rects = detector(gray_img, 1)
				landmarks = []
				text = ""
				for rect in rects:
					landmarks.append(np.array([[p.x, p.y] for p in predictor(gray_img, rect).parts()]))

				for landmark in landmarks:
					input_data = []
					face_landmark = []
					for i in range(len(landmark)):
						#cv2.drawMarker(img, (landmark[i][0], landmark[i][1]), (21, 255, 12))
						landmark_x = (landmark[i][0]-x)*100.00/width
						landmark_y = (landmark[i][1]-y)*100.00/height
						face_landmark.append(landmark_x)
						face_landmark.append(landmark_y)

					face_landmark = np.array(face_landmark).flatten()
					input_data.append(face_landmark)
					with graph.as_default():
						pred = model.predict(np.array(input_data))

					result_idx = np.argmax(pred[0])

					text = LABEL_INFO[result_idx] + ":" + str(int(pred[0][result_idx]*100.00)) + "%"
					print text

				#フォントの指定
				#font = cv2.FONT_HERSHEY_COMPLEX_SMALL
				font = cv2.FONT_HERSHEY_SIMPLEX
				#font = cv2.FONT_HERSHEY_PLAIN
				#文字の書き込み
				cv2.putText(img, text, (x, y), font, 0.5,(0,0,255))


		cv2.imshow("viewer", img)
		cv2.waitKey(1)

def main():
	camera()
	print len(LABEL_INFO.keys())

if __name__ == "__main__":
	main()


"""
cascade_path = "/home/macbook-air/face_identification/model/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_path)

model_path = "/home/macbook-air/face_identification/model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(model_path)
detector = dlib.get_frontal_face_detector()

trained_model_path = "/home/macbook-air/face_identification/model/weight.hdf5"
model = DNNModel().model
model.load_weights(trained_model_path)
graph = tf.get_default_graph()

test_image_path = "/home/macbook-air/face_identification/images/test_.jpeg"
result_image_path = "/home/macbook-air/face_identification/images/result_.jpeg"

img = cv2.imread(test_image_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
faces = cascade.detectMultiScale(gray_img, minSize=(30, 30))

if len(faces) != 0:
	for(x, y, width, height) in faces:
		cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 255), 1)
		rects = detector(gray_img, 1)
		landmarks = []
		for rect in rects:
			landmarks.append(np.array([[p.x, p.y] for p in predictor(gray_img, rect).parts()]))

		for landmark in landmarks:
			input_data = []
			face_landmark = []
			for i in range(len(landmark)):
				#cv2.drawMarker(img, (landmark[i][0], landmark[i][1]), (21, 255, 12))
				landmark_x = (landmark[i][0]-x)*100.00/width
				landmark_y = (landmark[i][1]-y)*100.00/height
				face_landmark.append(landmark_x)
				face_landmark.append(landmark_y)
			#face_landmark.append("\n")
			#face_landmarks.append(np.array(face_landmark).flatten())
			face_landmark = np.array(face_landmark).flatten()
			input_data.append(face_landmark)
			with graph.as_default():
				pred = model.predict(np.array(input_data))

			result_idx = np.argmax(pred[0])
			if result_idx == 0:
				#print "カルロス・ゴーン " + str(pred[0][result_idx])
				text = "Carlos:" + str(pred[0][result_idx])
				#text = "0"
			else:
				#print "Mr.ビーン " + str(pred[0][result_idx])
				text = "Rowan:" + str(pred[0][result_idx])
				#text = "1"

		#フォントの指定
		#font = cv2.FONT_HERSHEY_COMPLEX_SMALL
		font = cv2.FONT_HERSHEY_SIMPLEX
		#font = cv2.FONT_HERSHEY_PLAIN
		#文字の書き込み
		cv2.putText(img, text, (x, y), font, 0.5,(0,0,255))


#cv2.imshow("viewer", img)
cv2.imwrite(result_image_path, img)
"""