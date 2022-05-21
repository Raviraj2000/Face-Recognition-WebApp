import streamlit as st
from imageutils import saveImages
from person import addPerson
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
import numpy as np
from assets import loadModel, loadDB
import cv2
from deepface.commons import functions
from retinaface import RetinaFace
import os
from deepface.basemodels import Facenet
from video import whoisit
from web_rtc import play

@st.cache
model = loadModel()

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

play(model)




            

                





