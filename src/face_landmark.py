#!/usr/bin/env python
#coding:utf-8
import cv2
import dlib
import numpy as np
cap = cv2.VideoCapture(0)

cascade_path = "/home/macbook-air/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_path)

model_path = "/home/macbook-air/face_identification/model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(model_path)
detector = dlib.get_frontal_face_detector()

while True:
	_, img = cap.read()
	
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
			for points in landmark:
				cv2.drawMarker(img, (points[0], points[1]), (21, 255, 12))
		
		
		
		
	
	cv2.imshow("viewer", img)
	key = cv2.waitKey(1)
	
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()
