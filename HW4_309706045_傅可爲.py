#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
#讀取"train.csv"與"test.csv"並利用分割符號切割(sep='\t')
train = pd.read_csv('D:/NCTU/train.csv', sep='\t')
test = pd.read_csv('D:/NCTU/test.csv', sep='\t')
sam_sub = pd.read_csv('D:/NCTU/sample_submission.csv')
y_train = train["label"]
# y_train中會有一個”label”值，將之用0取代
y_train = y_train.replace('label',0,regex=True)
true = sam_sub["label"]


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
stopwords_list = ['a', 'the', 'and', 'it', 'to', 'for']
cv = CountVectorizer(stop_words =stopwords_list)
x_train = cv.fit_transform(train["text"])
x_test = cv.transform(test["text"])
print(x_train)
# print(x_test)


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
stopwords_list = ['a', 'the', 'and', 'it', 'to', 'for']
cv = TfidfVectorizer(stop_words =stopwords_list)
#Learn vocabulary and idf, return document-term matrix.
x_train = cv.fit_transform(train["text"])
#Transform documents to document-term matrix.
x_test = cv.transform(test["text"])
print(x_train)


# In[16]:


# XGBoost 模型
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score
label=y_train.astype('int')
#將Pandas data frame加入DMatrix中
dtrain = xgb.DMatrix(x_train, label=label)
# binary:logistic: 二元分類的邏輯回歸，輸出為機率
param = {'max_depth': 5, 'eta': 1, 'objective': 'binary:logistic'}
# boosting迭代次數設為10
num_round = 10
bst = xgb.train(param, dtrain,num_round)
dtest = xgb.DMatrix(x_test,label=true)
ypred1=bst.predict(dtest)
ypred1=np.where(ypred1 < 0.6, 0, 1)
# 計算confusion_matrix、accuracy、presion、recall、f1-score
C=confusion_matrix(true, ypred1)
print(C, '\n','accuracy: ', accuracy_score(true, ypred1))
precision=precision_score(true, ypred1)
recall=recall_score(true, ypred1)
f1_score=f1_score(true, ypred1)
print('precision:',precision, '\n', 'recall:', recall, '\n', 'f1-score:',f1_score)


# In[17]:


# Gradient Boosting 的分類法
from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier(random_state=0)
y_train_gbdt=y_train.astype('int')
clf=clf.fit(x_train, y_train_gbdt)
x_test_gbdt = x_test.astype('float')  
ypred2=clf.predict(x_test)
# 計算confusion_matrix、accuracy、presion、recall、f1-score
C=confusion_matrix(true, ypred2)
print(C, '\n','accuracy: ', accuracy_score(true, ypred2))
precision=precision_score(true, ypred2)
recall=recall_score(true, ypred2)
print('precision:',precision, '\n', 'recall:', recall, '\n', 'f1-score:', 2 * (precision * recall) / (precision + recall))


# In[19]:


# LightGBM是使用基於樹的學習算法的梯度上升框架
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
model = LGBMClassifier()
y_train_lgbm=y_train.astype('int')
model =model.fit(x_train,y_train_lgbm)
x_test_lgbm = x_test.astype('float')
ypred3=model.predict(x_test_lgbm)
# 計算confusion_matrix、accuracy、presion、recall、f1-score
C=confusion_matrix(true, ypred3)
print(C, '\n','accuracy: ', accuracy_score(true, ypred3))
precision=precision_score(true, ypred3)
recall=recall_score(true, ypred3)
print('precision:',precision, '\n', 'recall:', recall, '\n', 'f1-score:', 2 * (precision * recall) / (precision + recall))


# In[ ]:




