import faiss 
from deepface.basemodels import Facenet512


def loadDB(name):
    return faiss.read_index(name)
    
def saveDB(index, index_name):
    faiss.write_index(index, index_name)
    return


def loadModel():
    return Facenet512.loadModel()

