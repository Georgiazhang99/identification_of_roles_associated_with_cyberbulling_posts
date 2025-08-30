# Standard library imports
import os
import re
import math
import time
import ssl
import zipfile
import string
import pickle

# Data handling and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Imbalanced data handling
from imblearn.over_sampling import SMOTE

# Deep learning
import tensorflow as tf
import tensorflow_hub as hub

# Logging
from absl import logging

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')


#read CSV file and extract data
data_frame = pd.read_csv('ok.csv', header = None, encoding = 'utf-8')
trainText = data_frame[0]
trainLabel = data_frame[1]

# Fix SSL issue for macOS
ssl._create_default_https_context = ssl._create_unverified_context
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

# Sentence Embeddings - Universal Encoder from TF-Hub Embeddings- pretrained spacy embeddings
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)

# Reduce logging output.
logging.set_verbosity(logging.ERROR)
stemmer = WordNetLemmatizer()

def pre_process(text):
    # Remove all the special characters
    text = re.sub(r'\W', ' ', str(text))
    # remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)
    # Converting to Lowercase
    text = text.lower()
    # Lemmatization
    text = text.split()
    text = [stemmer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text

train = trainText.apply(pre_process)
trainEmbeddings = embed(train)

# apply oversampling on embeddings
sm = SMOTE(k_neighbors=1) 
trainEmbeddings, trainLabel = sm.fit_resample(trainEmbeddings, trainLabel.ravel())

print("counts of label 'Harasser': {}".format(sum(trainLabel== 'Harasser'))) 
print("counts of label 'Victim': {} \n".format(sum(trainLabel == 'Victim')))

X_train, X_test, y_train, y_test = train_test_split(trainEmbeddings, trainLabel, test_size=0.1,random_state=109)

# fit model no training data
classifier = LogisticRegression()

#Train the model using the training sets
classifier.fit(X_train, y_train, sample_weight = None)

#Predict the response for test dataset
y_pred = classifier.predict(X_test)


print("Identifying Harasser Metrics")
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, pos_label = 'Harasser'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, pos_label = 'Harasser'))

#measure F1 score
print("F1 score: ", metrics.f1_score(y_test, y_pred, labels = None, pos_label = 'Harasser', average = 'binary', sample_weight = None))

print("")
#victim scores
print("Identifying Victim Metrics")


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, pos_label = 'Victim'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, pos_label = 'Victim'))

#measure F1 score
print("F1 score: ", metrics.f1_score(y_test, y_pred, labels = None, pos_label = 'Victim', average = 'binary', sample_weight = None))
print("Weighted scores")
print("weighted F1:", metrics.f1_score(y_test, y_pred, average='weighted'))
print("weighted Precision:", metrics.precision_score(y_test, y_pred, average='weighted'))
print("weighted Recall:", metrics.recall_score(y_test, y_pred, average='weighted'))
print("Error rate:", 1-metrics.accuracy_score(y_test, y_pred))

