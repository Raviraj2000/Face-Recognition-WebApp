import os
from PIL import Image
import streamlit as st

def saveImages(name, images):
    st.write("Save Images")
    dir_path = os.path.join("images", name)
    st.write(dir_path)
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)

    for i, image in enumerate(images):
        img = Image.open(image)
        path = os.path.join(dir_path, name + "_" + str(i) + ".jpg")
        img.save(path, 'JPEG')
    
    return 

def loadImages(name):
    path = os.path.join("images", name)
    images = []
    for (root,dirs,files) in os.walk(path):
        for file in files:
            exact_path = root+"/"+file
            print(root+"/"+file)
            images.append(exact_path)
    return images