import cv2
import streamlit as st
from deepface.commons import functions
from assets import loadModel
import os
import numpy as np
from retinaface import RetinaFace
from PIL import Image

def whoisit(index, names):
    model = loadModel()

    st.title("Webcam Live Feed")
    run = True

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        print("running")
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = RetinaFace.detect_faces(img_path = frame)
        
        for face in faces:
            FRAME_WINDOW.image(frame)
            x1, y1, width, height = faces[face]['facial_area']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            face = frame[y1:y2, x1:x2]


            print("got co-ordinates")
            image = np.array(face)
            print('got face')
            image = functions.preprocess_face(img = image,
                                            target_size = (160,160),
                                            detector_backend='opencv',
                                            enforce_detection=False)
                                            
            image = model.predict(image)
            
            
            _, neighbors = index.search(image, 1)
            
            i = neighbors[0][0]
            

            cv2.rectangle(frame, (x1,y1), (width, height), (255,0,255))
            cv2.putText(frame, os.path.basename(names[i]), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            print("finished")

        FRAME_WINDOW.image(frame)

    return
