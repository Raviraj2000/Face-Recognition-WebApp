from streamlit_webrtc import webrtc_streamer
import av
import cv2
from assets import loadModel, loadDB
from retinaface import RetinaFace
import numpy as np
from deepface.commons import functions
import os


class VideoProcessor:
    def recv(self, frame):
        model = loadModel()
        index = loadDB('database/faces.index')
        data = np.load('database/names.npz', allow_pickle=True)
        names = data['arr_0'].tolist()

        img = frame.to_ndarray(format="bgr24")
        faces = RetinaFace.detect_faces(img_path = img)
        
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
                                            
            image = model.predict(image)
            
            
            _, neighbors = index.search(image, 1)
            
            i = neighbors[0][0]
            

            cv2.rectangle(frame, (x1,y1), (width, height), (255,0,255))
            cv2.putText(frame, os.path.basename(names[i]), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            print("finished")
        

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="detector", video_processor_factory=VideoProcessor)