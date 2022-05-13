import streamlit as st
from imageutils import saveImages
from person import addPerson
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
from assets import loadModel, loadDB
import cv2
from deepface.commons import functions
from retinaface import RetinaFace
import os
from deepface.basemodels import Facenet


model = Facenet.InceptionResNetV2(dimension = 512)
model.load_weights('facenet512_weights.h5')

class VideoProcessor:
    def __init__(self):
        self.model = model
        self.index = loadDB('database/faces.index')
        data = np.load('database/names.npz', allow_pickle=True)
        self.names = data['arr_0'].tolist()

        return

    def recv(self, frame):

        frame = frame.to_ndarray(format="bgr24")
        faces = RetinaFace.detect_faces(img_path = frame)

        for face in faces:
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
                                            
            image = self.model.predict(image)
            
            
            _, neighbors = self.index.search(image, 1)
            
            i = neighbors[0][0]
            

            cv2.rectangle(frame, (x1,y1), (width, height), (255,0,255))
            cv2.putText(frame, os.path.basename(self.names[i]), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            print("finished")

        return av.VideoFrame.from_ndarray(frame, format='bgr24')


name = st.text_input("Enter Name")

if name:
    st.write("Hello ", name)
    st.subheader("Please upload your photos")

    images = st.file_uploader('Add a person', type=['png','jpg'], 
                            accept_multiple_files=True, 
                            key='userfiles')

    check = st.button("Upload Files", key='uploadfiles')

    if check:
        if images == None:
            st.error("Please Upload Files in png or jpg Format")
        else:     
            saveImages(name, images)
            result= addPerson(name, model)
            st.write(result)

webrtc_streamer(key="key", video_processor_factory=VideoProcessor)


            

                





