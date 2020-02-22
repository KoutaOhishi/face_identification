#!/usr/bin/env python
#coding:utf-8
import os
import cv2
import dlib
import numpy as np

REPOSITORY_PATH = str("/".join(os.path.abspath(__file__).split("/")[0:-2]))
CASCADE_PATH = REPOSITORY_PATH + "/model/haarcascade_frontalface_alt.xml"
MODEL_PATH = REPOSITORY_PATH + "/model/shape_predictor_68_face_landmarks.dat"

SAVE_FILE_DIR = REPOSITORY_PATH + "/dataset/"
SAVE_FILE_NAME = "test.csv"

DATA_NUM = 100

def camera():
	cap = cv2.VideoCapture(0)
	cascade = cv2.CascadeClassifier(CASCADE_PATH)
	predictor = dlib.shape_predictor(MODEL_PATH)
	detector = dlib.get_frontal_face_detector()

	face_landmarks = []

	while True:
		_, img = cap.read()

		gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		faces = cascade.detectMultiScale(gray_img, minSize=(30, 30))


		print len(face_landmarks)
		if len(faces) != 0:
			for(x, y, width, height) in faces:
				cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 255), 1)
				rects = detector(gray_img, 1)
				landmarks = []
				for rect in rects:
					landmarks.append(np.array([[p.x, p.y] for p in predictor(gray_img, rect).parts()]))

				for landmark in landmarks:
					face_landmark = []
					for i in range(len(landmark)):
						cv2.drawMarker(img, (landmark[i][0], landmark[i][1]), (21, 255, 12))
						landmark_x = (landmark[i][0]-x)*100.00/width
						landmark_y = (landmark[i][1]-y)*100.00/height
						face_landmark.append(landmark_x)
						face_landmark.append(landmark_y)
					#face_landmark.append("\n")
					face_landmarks.append(np.array(face_landmark).flatten())

		if len(face_landmarks) > DATA_NUM:
			break

		cv2.imshow("viewer", img)
		key = cv2.waitKey(1)

	cap.release()
	cv2.destroyAllWindows()

	print "finish"
	print len(face_landmarks)
	print len(face_landmarks[0])
	np_dataset = np.array(face_landmarks)
	np.savetxt(SAVE_FILE_DIR+SAVE_FILE_NAME, np_dataset)



def image_file():
	face_landmarks = []

	image_file_dir = REPOSITORY_PATH + "/images/rowan/"

	for n in range(10):
		#image_file_name = "carlos"+str(n)+".jpeg"
		image_file_name = "rowan"+str(n)+".jpeg"
		raw_img = cv2.imread(image_file_dir+image_file_name)

		original_width, original_height = raw_img.shape[:2]
		multiple_list = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
		for m in multiple_list:
			size = (int(original_height*m), int(original_width*m))
			img = cv2.resize(raw_img, size)

			gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			faces = cascade.detectMultiScale(gray_img)

			if len(faces) != 0:
				for(x, y, width, height) in faces:
					cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 255), 1)
					rects = detector(gray_img, 1)
					landmarks = []
					for rect in rects:
						landmarks.append(np.array([[p.x, p.y] for p in predictor(gray_img, rect).parts()]))

					for landmark in landmarks:
						face_landmark = []
						for i in range(len(landmark)):
							cv2.drawMarker(img, (landmark[i][0], landmark[i][1]), (21, 255, 12))
							landmark_x = (landmark[i][0]-x)*100.00/width
							landmark_y = (landmark[i][1]-y)*100.00/height
							face_landmark.append(landmark_x)
							face_landmark.append(landmark_y)
						#face_landmark.append("\n")
						face_landmarks.append(np.array(face_landmark).flatten())

	print "finish"
	print len(face_landmarks)
	print len(face_landmarks[0])
	np_dataset = np.array(face_landmarks)
	np.savetxt(SAVE_FILE_DIR+SAVE_FILE_NAME, np_dataset)


if __name__ == "__main__":
	camera()

