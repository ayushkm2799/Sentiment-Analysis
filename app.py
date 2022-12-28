#https://docs.pylonsproject.org/projects/pyramid/en/latest/

from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix,classification_report
import re
import numpy as np
import pandas as pd
from helper import Model
import pickle
# In[6]:


import __main__
setattr(__main__, "Model", Model)

model1 = Model()
model1.load()

from tensorflow.keras.models import load_model
model = load_model('model.h5')
#model = tf.keras.models.load_model('model.h5')

with open('tokenizer.pickle', 'rb') as handle:
	tokenizer1 = pickle.load(handle)

@view_config(route_name="hello")
def hello_world(request):

    return Response("Server is Up and running")
    
def predict_using_DL(text):
    twt = [text]
    max_fatures = 2000
    twt = tokenizer1.texts_to_sequences(twt)
    twt = pad_sequences(twt, maxlen=32, dtype='int32', value=0)
    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
    if(np.argmax(sentiment) == 0):
        prediction="negative"
    elif (np.argmax(sentiment) == 1):
        prediction="positive"
    return prediction
 

def predict_using_ML(text):
    out=model1.predict(text)
    return  out[0]

@view_config(route_name="predict", renderer="json", request_method="POST", openapi=True)
def predict(request):
    text = request.openapi_validated.body["text"]
    twt=[text]
    max_fatures = 2000
    twt = tokenizer1.texts_to_sequences(twt)
    twt = pad_sequences(twt, maxlen=32, dtype='int32', value=0)
    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
    if(np.argmax(sentiment) == 0):
        prediction="negative"
    elif (np.argmax(sentiment) == 1):
        prediction="positive"
    return {
        'prediction': prediction
    }

if __name__ == "__main__":
    with Configurator() as config:
        config.include("pyramid_openapi3")
        config.pyramid_openapi3_spec("apidocs.yaml")
        config.pyramid_openapi3_add_explorer()
        config.add_route("predict", "/predict")
        config.add_route("hello", "/")
        config.scan(".")
        app = config.make_wsgi_app()
    print("Server started at http://0.0.0.0:8080/")
    print("Swagger UI documentation can be found at http://0.0.0.0:8080/docs/")
    server = make_server('0.0.0.0', 8080, app)
    server.serve_forever()
