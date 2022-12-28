#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np 
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import pickle


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support



class Model:
    def __init__(self, datafile = "airline_sentiment_analysis.csv"):
        self.data = pd.read_csv(datafile)
        self.porter = PorterStemmer()
        self.tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
        self.kernel = 'rbf'
        self.degree=3
        self.pred_model=None

    def split(self, test_size):
        y = self.data['airline_sentiment']
        X = self.data['text']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    def tokenizer(self, text):
        return text.split()
    
    def tokenizer_porter(self, text):
        return [self.porter.stem(word) for word in text.split()]
    
    def preprocessor(self, text):
        # Remove HTML markup
        text = re.sub('<[^>]*>', '', text)
        
        text = re.sub(r'https?:\/\/\S+', '' , text)
        text = re.sub(r'\w*\@*\w*\.(com)\w*', '', text)
        text = re.sub(r'^(emailmailto:)\w*\.*\w+\@*\w+\.com',' ',text)
        # Save emoticons for later appending
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        # Remove any non-word character and append the emoticons,
        # removing the nose character for standarization. Convert to lower case
        text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))
        return text
    
    def logisticregression(self):
        param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__tokenizer': [self.tokenizer, self.tokenizer_porter],
               'vect__preprocessor': [None, self.preprocessor],
               'clf__penalty': ['l2'],
               'clf__C': [1.0]},
              ]

        lr_pipe = Pipeline([('vect', self.tfidf),
                     ('clf', LogisticRegression(random_state=0))])

        lr_clf = GridSearchCV(lr_pipe, param_grid,scoring='accuracy',cv=5,verbose=1,n_jobs=-1)
        return lr_clf
    
    def support_vector_machine(self, kernal, degree):
        param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__tokenizer': [self.tokenizer, self.tokenizer_porter],
               'vect__preprocessor': [None, self.preprocessor]},
              ]

        cvm_pipe = Pipeline([('vect', self.tfidf),
                     ('clf',  svm.SVC(C=9.0,kernel=self.kernel, degree=self.degree,random_state=42))])

        svm_clf = GridSearchCV(cvm_pipe, param_grid,scoring='accuracy',cv=5)
        return svm_clf
    
    def fit(self,model):
        if model == 'svm':
            clf = self.support_vector_machine('linear',3)
            self.model = clf.fit(self.X_train, self.y_train)
        if model == 'lr':
            clf = self.logisticregression()
            self.model = clf.fit(self.X_train, self.y_train)
        self.pred_model = self.model.best_estimator_
        return self.model.best_estimator_
    
    def accuracy(self,clf):
        print('Accuracy in test: %.3f' % clf.score(self.X_test, self.y_test))
    
    def predict(self, text):
        text = self.preprocessor(text)
        return self.pred_model.predict([text])

    def load(self, filename='test_data1.pkl'):
        with open(filename, 'rb') as f:
            self.pred_model = pickle.load(f)
    def save(self, filename='test_data1.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.pred_model, f)




