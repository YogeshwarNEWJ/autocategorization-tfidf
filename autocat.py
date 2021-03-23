# -*- coding: utf-8 -*-

# libraries
import pandas as pd
import numpy as np
import nltk
import re
import pickle
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from autocat_1 import pre_process

# data
data = pd.read_csv('..\\combine_dataframes\\refined_golden_data.csv')

# select columns
refined_data = data[['Permalink', 'title', 'description', 'Primary_Category_1_grouped']]

refined_data = refined_data[refined_data['Primary_Category_1_grouped'] != 'rare']

# clean
corpus = []
for sent in refined_data['title']:
    title = re.sub(r'\W',' ', sent)
    title = title.lower()
    title = re.sub(r'\s+[a-z]\s+',' ', title)
    title = re.sub(r'^\s+','', title)
    corpus.append(title)
    
# BOW    
vercorizer = CountVectorizer(max_features=3000, min_df = 20, max_df = 0.7,\
                             stop_words = stopwords.words('english'))
X = vercorizer.fit_transform(corpus).toarray()

# tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()
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
model = CatBoostClassifier(iterations=3000, learning_rate=0.01,\
                           l2_leaf_reg=3.5, depth=12,\
                           custom_metric = 'F1', rsm=0.98,\
                           use_best_model=True,random_seed=42) 
                            #test_type = 'GPU'

model.fit(X_train, y_train, plot = True, eval_set=(X_test, y_test))


# validation
preds_class = model.predict(X_test)
val_data = pd.DataFrame({'actual': y_test, 'pred': list(np.reshape(preds_class, (1,576)))[0]})






# shap_values = model.get_feature_importance(Pool(X, y), type='ShapValues')