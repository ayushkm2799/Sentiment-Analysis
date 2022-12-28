


import uvicorn
from fastapi import FastAPI
from Sentiment_class import sentiment_request
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


import pickle
description = """
Airline Sentiment Analysis API helps you for prediction of whether the text is postitve sentiment 😁 or negative 
sentiment 😡.

## index

WELCOME page 

## Predict

Analyzes a piece of text and determines whether the text has a positive or negative sentiment

"""

tags_metadata = [
    
    {
        "name": "predict",
        "description": "Analyzes a piece of text and determines whether the text has a positive or negative sentiment. Use either 'DL' or 'ML' as a model type, any other model type cause error",
        "externalDocs": {
            "description": "Items external docs",
            "url": "https://fastapi.tiangolo.com/",
        },
    },
]
print('HELLO',flush = True)


sentiment_api = FastAPI(title="Airline Sentiment Analysis",description=description,openapi_tags=tags_metadata)
#to avoid pickle attribute error

import __main__
setattr(__main__, "Model", Model)

model1 = Model()
model1.load()

from tensorflow.keras.models import load_model
model = load_model('model.h5')
#model = tf.keras.models.load_model('model.h5')

with open('tokenizer.pickle', 'rb') as handle:
	tokenizer1 = pickle.load(handle)

print('HELLO',flush = True)

@sentiment_api.get('/')
def index():
    return {'message': 'Hello and Welcome to sentiment classification api'}

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

@sentiment_api.post('/predict', tags=["predict"])
def predict_sentiment_DL(request:sentiment_request):
    if request.model_type.lower() == 'dl':
        return { "Prediction": predict_using_DL(request.text)}
    elif request.model_type.lower() == 'ml':
        return { "Prediction": predict_using_ML(request.text)}
    else:
        return { "ERROR": "Use either 'DL' or 'ML' as a model type"}



# In[13]:


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.2', port=8000)

