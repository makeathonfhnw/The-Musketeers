#!/usr/bin/env python
# coding: utf-8

# In[572]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[573]:


cd


# In[574]:


cd \Users\glisi\Desktop\MakeAthon


# In[575]:


df = pd.read_csv('modellMitScore.csv', sep = ';')


# In[576]:


df.head()


# In[577]:


df.shape


# In[578]:


df.columns


# In[579]:


df.dtypes


# In[580]:


df.shape


# In[581]:


df['score'].value_counts()


# In[582]:


df['score'].value_counts(normalize=True) 


# In[583]:


from sklearn.model_selection import train_test_split
X = df.drop('score',axis=1)
y = df[['score']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=42)


# In[584]:


df.info()


# In[585]:


target = list(df['score'].unique())
feature_names = list(X.columns)


# In[586]:


from sklearn import tree
import graphviz


# In[587]:


from sklearn.tree import DecisionTreeClassifier
clf_model = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)   
clf_model.fit(X_train,y_train)


# In[588]:


y_predict = clf_model.predict(X_test)


# In[589]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(y_test,y_predict)


# In[590]:


df.shape


# In[591]:


target = list(df['score'].unique())
feature_names = list(X.columns)


# In[592]:


from sklearn.tree import export_text
r = export_text(clf_model, feature_names=feature_names)
print(r)


# In[593]:


from sklearn.model_selection import train_test_split
X = df.drop('score',axis=1)
y = df[['score']]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7,random_state=42)


# In[594]:


target = list(df['score'].unique())
feature_names = list(X.columns)


# In[595]:


from sklearn import tree
import graphviz


# In[608]:


from sklearn.tree import DecisionTreeClassifier
clf_model = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=20, min_samples_leaf=5, ccp_alpha=0.0015)   
clf_model.fit(X_train,y_train)


# In[609]:


y_predict = clf_model.predict(X_test)


# In[610]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(y_test,y_predict)


# In[611]:


cfm = confusion_matrix(y_test, y_predict)
sns.heatmap(cfm, annot=True)


# In[612]:


target = list(df['score'].unique())
feature_names = list(X.columns)


# In[613]:


from sklearn.tree import export_text
r = export_text(clf_model, feature_names=feature_names)
print(r)


# In[620]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(X_train, y_train)


# In[621]:


pred_cv = model.predict(X_test)
accuracy_score(y_test,pred_cv)


# In[622]:


from sklearn.linear_model import LinearRegression
model_reg = LinearRegression()
reg = model_reg.fit(X_train, y_train)
reg.coef_


# In[623]:


X = df.drop('score',1)
y = df.score


# In[624]:


X = pd.get_dummies(X)
train=pd.get_dummies(df)
test=pd.get_dummies(df)


# In[625]:


eb = pd.read_csv('revisedModelWithEligibility.csv', sep = ';')
eb.head()


# In[626]:


eb.shape


# In[627]:


from sklearn.model_selection import train_test_split
X = eb.drop('Eligible',axis=1)
y = eb[['Eligible']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=42)


# In[628]:


target = list(eb['Eligible'].unique())
feature_names = list(X.columns)


# In[629]:


from sklearn import tree
import graphviz


# In[630]:


from sklearn.tree import DecisionTreeClassifier
clf_model = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=15, min_samples_leaf=5, ccp_alpha=0.0015)   
clf_model.fit(X_train,y_train)


# In[631]:


y_predict = clf_model.predict(X_test)


# In[632]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(y_test,y_predict)


# In[633]:


from sklearn.tree import export_text
r = export_text(clf_model, feature_names=feature_names)
print(r)


# In[634]:


cfm = confusion_matrix(y_test, y_predict)
sns.heatmap(cfm, annot=True)


# In[635]:


from sklearn.linear_model import LinearRegression


# In[639]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(X_train, y_train)


# In[640]:


pred_cv = model.predict(X_test)
accuracy_score(y_test,pred_cv)


# In[641]:


from sklearn.linear_model import LinearRegression
model_reg = LinearRegression()
reg = model_reg.fit(X_train, y_train)
reg.coef_


# In[ ]:




