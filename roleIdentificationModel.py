import zipfile
import pandas as pd
import re
import math
import time
from sklearn.model_selection import train_test_split
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

#logisticregression Classifier
from sklearn import datasets
import pandas as pd
import numpy as np
import nltk
import re
import string

#WE ARE USING SENTENCE EMBEDDINGS- Universal Encoder from TF-Hub Embeddings- pretrained spacy embeddings
#help from https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
#module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" uses this module
from sklearn import datasets
import pandas as pd
import numpy as np
import nltk
import re
import string


#using TF-HUB Universal Encoder
#Install the latest Tensorflow Hub version of the Universal Sentence Encoder
!pip3 install --quiet "tensorflow>=1.7"
!pip3 install --quiet tensorflow-hub
!pip3 install --quiet seaborn

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
import seaborn as sns

#upload dataset
from google.colab import files
uploaded = files.upload()

#check data is there
!ls

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])

print(embeddings)

#read CSV file and extract data
data_frame = pd.read_csv('ok.csv', header = None, encoding = 'utf-8')
trainText = data_frame[0]
trainLabel = data_frame[1]

#@title Load the Universal Sentence Encoder's TF Hub module
from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
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

#SMOTE FOR OVERSAMPLING
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
trainEmbeddings, trainLabel = sm.fit_sample(trainEmbeddings, trainLabel.ravel()) 


print("counts of label 'Harasser': {}".format(sum(trainLabel== 'Harasser'))) 
print("counts of label 'Victim': {} \n".format(sum(trainLabel == 'Victim')))

print("counts of label 'Harasser': {}".format(sum(trainLabel== 'Harasser'))) 
print("counts of label 'Victim': {} \n".format(sum(trainLabel == 'Victim')))

print(len(trainLabel))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainEmbeddings, trainLabel, test_size=0.1,random_state=109)

from sklearn.linear_model import LogisticRegression

# fit model no training data
classifier = LogisticRegression()

#Train the model using the training sets
classifier.fit(X_train, y_train, sample_weight = None)

#Predict the response for test dataset
y_pred = classifier.predict(X_test)


from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print("Identifying Harasser Metrics")
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, pos_label = 'Harasser'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, pos_label = 'Harasser'))

#measure F1 score
from sklearn.metrics import f1_score
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
from sklearn.metrics import f1_score
print("F1 score: ", metrics.f1_score(y_test, y_pred, labels = None, pos_label = 'Victim', average = 'binary', sample_weight = None))
print("Weighted scores")
print("weighted F1:", metrics.f1_score(y_test, y_pred, average='weighted'))
print("weighted Precision:", metrics.precision_score(y_test, y_pred, average='weighted'))
print("weighted Recall:", metrics.recall_score(y_test, y_pred, average='weighted'))
print("Error rate:", 1-metrics.accuracy_score(y_test, y_pred))

