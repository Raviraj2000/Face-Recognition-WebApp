from deepface.commons import functions
import streamlit as st
from imageutils import loadImages
import numpy as np
import faiss
from assets import loadModel

def addPerson(name):
    #index = loadDB('database/faces.index')
    model = loadModel()
    st.write("database and model loaded")

    facial_images = loadImages(name)
    st.write(facial_images)
    
    data = np.load('database/names.npz', allow_pickle=True)
    names = data['arr_0'].tolist()
    data = np.load('database/representations.npz', allow_pickle=True)
    representations = data['arr_0'].tolist()

    with st.spinner('Calculating embeddings....'):
        for face in facial_images:
            
            img = functions.preprocess_face(img = face, target_size = (160,160), 
                                            detector_backend='retinaface')
            embedding = model.predict(img)[0, :]

            representation = []
            representation.append(face)
            names.append(face)
            representation.append(embedding)

            representations.append(representation)

    st.write("Embeddings generated")
    
    embeddings = []
    for i in range(0, len(representations)):
        embedding = representations[i][1]
        embeddings.append(embedding)
    embeddings = np.array(embeddings, dtype='f')

    index = faiss.IndexFlatIP(512)
    faiss.normalize_L2(embeddings)

    index.add(embeddings)
    faiss.write_index(index, "database/faces.index")
    np.savez('database/names.npz', names)
    return str(name + " added to database")