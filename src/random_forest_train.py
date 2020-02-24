#!/usr/bin/env python
#coding:utf-8
import os
import numpy as np
import cv2
import dlib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV



REPOSITORY_PATH = str("/".join(os.path.abspath(__file__).split("/")[0:-2]))
CASCADE_PATH = REPOSITORY_PATH + "/model/haarcascade_frontalface_alt.xml"
MODEL_PATH = REPOSITORY_PATH + "/model/shape_predictor_68_face_landmarks.dat"
TRAIN_DATA_DIR = REPOSITORY_PATH + "/dataset/lab_members/"
LABEL_INFO = []


def create_train_data():
    files = os.listdir(TRAIN_DATA_DIR)
    label_num = len(files)
    landmarks = []
    labels = []
    label_info = []

    for i, file in enumerate(files):
        with open(TRAIN_DATA_DIR+file, "r") as f:
            lines = f.read().split("\n")
        f.close()

        for line in lines:
            marks= line.split(" ")
            landmarks.append(np.array(marks).flatten())
            labels.append(i)

        #LABEL_INFO.append(file.split(".csv")[0])
        label_info.append(file.split(".csv")[0])

    return landmarks, labels, label_info

def train():
    x, y, label_info = create_train_data()

    clf = RandomForestClassifier(max_depth=10, n_estimators=400, bootstrap=True, criterion="gini")
    clf.fit(x,y)

    save_tree_graph(clf)

def save_tree_graph(forest):
    from sklearn import tree
    for i,val in enumerate(forest.estimators_):
        tree.export_graphviz(forest.estimators_[i], out_file=REPOSITORY_PATH+'/tree_%d.dot'%i)

def param_search():
    param_grid = {#"max_depth": [2,3, None],
              "n_estimators":[50,100,200,300,400,500],
              #"max_features": [1, 3, 10],
              #"min_samples_split": [2, 3, 10],
              #"min_samples_leaf": [1, 3, 10],
              #"bootstrap": [True, False],
              #"criterion": ["gini", "entropy"]
              }

    forest_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                 param_grid = param_grid,
                 scoring="accuracy",  #metrics
                 cv = 3,              #cross-validation
                 n_jobs = 4)          #number of core

    x, y, label_num = create_train_data()
    forest_grid.fit(x, y) #fit

    forest_grid_best = forest_grid.best_estimator_ #best estimator
    print("Best Model Parameter: ",forest_grid.best_params_)



def camera():
    x, y, label_info = create_train_data()

    clf = RandomForestClassifier()
    clf.fit(x,y)

    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    predictor = dlib.shape_predictor(MODEL_PATH)
    detector = dlib.get_frontal_face_detector()

    while True:
        _, img = cap.read()

        start_time = time.time()

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

                    pred_idx = clf.predict(input_data)[0]
                    #print pred[0]
                    text = str(label_info[pred_idx])
                    #print text

                #フォントの指定
                #font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                font = cv2.FONT_HERSHEY_SIMPLEX
                #font = cv2.FONT_HERSHEY_PLAIN
                #文字の書き込み
                cv2.putText(img, text, (x, y), font, 0.5,(0,0,255))

        process_time = time.time() - start_time
        fps = 1.0 / process_time
        print fps
        cv2.imshow("viewer", img)
        cv2.waitKey(1)


def main():
    #train()
    #param_search()
    camera()

if __name__ == "__main__":
    main()