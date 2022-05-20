import faiss 
from deepface.basemodels import Facenet512


def loadDB(name):
    return faiss.read_index(name)
    
def saveDB(index, index_name):
    faiss.write_index(index, index_name)
    return


def loadModel():
    model = Facenet.InceptionResNetV2(dimension = 512)
    model.load_weights('facenet512_weights.h5')
    #model = Facenet512.loadModel()
    return model

