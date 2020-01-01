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

face_landmarks = []

#image_file_dir = "/home/macbook-air/face_identification/images/carlos/"
image_file_dir = "/home/macbook-air/face_identification/images/rowan/"

save_file_dir = "/home/macbook-air/face_identification/dataset/"
#save_file_path = save_file_dir + "carlos.csv"	
save_file_path = save_file_dir + "rowan.csv"	

for n in range(10):
	#image_file_name = "carlos"+str(n)+".jpeg"
	image_file_name = "rowan"+str(n)+".jpeg"	
	raw_img = cv2.imread(image_file_dir+image_file_name)
	#_, img = cap.read()
	#original_height, original_width = raw_img.shape[:2]
	original_width, original_height = raw_img.shape[:2]
	multiple_list = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
	for m in multiple_list:	
		size = (int(original_height*m), int(original_width*m))
		img = cv2.resize(raw_img, size)

		gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		faces = cascade.detectMultiScale(gray_img)

		flag = False

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
					flag = True


		if flag == False:
			print n
			print m
			print "---"


		
	
	

		
		
	
		#cv2.imshow("viewer", img)
		#key = cv2.waitKey(100)
	
	#if key == 27:
	#	break

cap.release()
cv2.destroyAllWindows()

print "finish"
print len(face_landmarks)
print len(face_landmarks[0])
np_dataset = np.array(face_landmarks)
np.savetxt(save_file_path, np_dataset)

