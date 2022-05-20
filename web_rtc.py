import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
import cv2
from assets import loadModel, loadDB
from retinaface import RetinaFace
import numpy as np
from deepface.commons import functions
import os
import threading
from typing import Union

def play():

    class VideoProcessor(VideoTransformerBase):

        def __init__(self) -> None :
            self.model = loadModel()
            self.index = loadDB('database/faces.index')
            data = np.load('database/names.npz', allow_pickle=True)
            self.names = data['arr_0'].tolist()

        def recv(self, frame):

            frame = frame.to_ndarray(format="bgr24")
            faces = None    
            
            faces = RetinaFace.detect_faces(img_path = frame)
            if faces == None:
                return av.VideoFrame.from_ndarray(frame, format='bgr24')
            
          
            for face in faces:
                x1, y1, width, height = faces[face]['facial_area']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                face = frame[y1:y2, x1:x2]



                image = np.array(face)

                image = functions.preprocess_face(img = image,
                                                target_size = (160,160),
                                                detector_backend='opencv',
                                                enforce_detection=False)
                                                
                image = self.model.predict(image)
                
                
                _, neighbors = self.index.search(image, 1)
                
                i = neighbors[0][0]
                

                cv2.rectangle(frame, (x1,y1), (width, height), (255,0,255))
                cv2.putText(frame, os.path.basename(self.names[i]), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
 
            return av.VideoFrame.from_ndarray(frame, format='bgr24')

    webrtc_streamer(key="detector", video_processor_factory=VideoProcessor,
                    rtc_configuration={  # Add this line
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    })
    


    return
