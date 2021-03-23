# -*- coding: utf-8 -*-

# libraries
import pandas as pd
import numpy as np
import nltk
import re
import pickle
import heapq
from string import punctuation
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from catboost import CatBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE

# data
data = pd.read_csv('..\\combine_dataframes\\refined_golden_data.csv')
refined_data = data[['title', 'description', 'Primary_Category_1_grouped']]

# filter categories
refined_data = refined_data[refined_data['Primary_Category_1_grouped'] != 'rare']
refined_data = refined_data[refined_data['Primary_Category_1_grouped'] != 'others']

# preprocessing 
corpus = []

def pre_process(description):    
    str_punct = '[!"#$%&\'()*+,-./:;<=>?@\^_`{|}~]'
    description = description.lower()
    description = re.sub(str_punct, '', description)
    description = re.sub(r'\s+[a-z]\s+',' ', description)    
    description = re.sub(r'^\s+','', description)
    description = re.sub(r'[0-9]', '', description)
    return description

for sent in refined_data['description']:
    sent = pre_process(sent)    
    corpus.append(sent)
    
# creating the histogram
word2count = {}
for message in corpus:
    words = nltk.word_tokenize(message)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1      

# stopwords
stop = open('final_stopwords.txt', encoding="utf8")
stopwords=[]
for x in stop:
    stopwords.append(str(x))
    
stopwords = [i.split('\n')[0] for i in stopwords]
[word2count.pop(stwd) for stwd in stopwords if stwd in list(word2count.keys())]

# frequent 3000 words
freq_words = heapq.nlargest(3000, word2count, key=word2count.get)

# IDF matrix
word_idfs = {}
for word in freq_words:
    doc_count = 0
    for desc in refined_data['description']:
        if word in nltk.word_tokenize(desc):
            doc_count += 1
    if doc_count > 0:
        word_idfs[word] = np.log((len(refined_data)/doc_count)+1)
    else:
        word_idfs[word] = 0
        
# TF matrix
tf_matrix = {}
for word in freq_words:
    doc_tf = []
    for desc in refined_data['description']:
        frequency = 0
        for w in nltk.word_tokenize(desc):
            if w == word:
                frequency += 1
        tf_word = frequency/len(nltk.word_tokenize(desc))
        doc_tf.append(tf_word)
    tf_matrix[word] = doc_tf
    
# TF-IDF Calculaton
tfidf_matrix = []
for word in tf_matrix.keys():
    tfidf = []
    for value in tf_matrix[word]:
        score = value * word_idfs[word]
        tfidf.append(score)
    tfidf_matrix.append(tfidf)
    
X = np.asarray(tfidf_matrix)

X = X.T
y = list(refined_data['Primary_Category_1_grouped'])

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y,\
                                                    test_size = 0.2,\
                                                    random_state = 51)

# imbalanced data hadeling
sm = SMOTE(random_state = 51)
#X_train_res, y_train_res = 
sm.fit(X_train, y_train)

# modeling
model = CatBoostClassifier(iterations = 20000, learning_rate=0.01,\
                           depth=5, custom_metric = 'F1',\
                           use_best_model=True,random_seed=42) 
                            #test_type = 'GPU'

model.fit(X_train, y_train, plot = True, eval_set=(X_test, y_test))


# validation
preds_class = model.predict(X_test)
val_data = pd.DataFrame({'actual': y_test, 'pred': list(np.reshape(preds_class, (1,444)))[0]})

        