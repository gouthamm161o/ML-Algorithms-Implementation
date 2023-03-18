#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,roc_auc_score,make_scorer
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# In[8]:


pd.set_option("display.max_columns",None)


# In[9]:


data=pd.read_csv("Downloads\diabetes.csv")


# In[10]:


data


# In[12]:


data.dtypes


# In[13]:


data.isnull()


# In[15]:


data.describe().T


# In[16]:


data["Outcome"].value_counts()


# In[18]:


y=data["Outcome"]
x=data.drop(['Outcome'],axis=1)


# In[19]:


sns.pairplot(data, hue="Outcome",
             diag_kws=dict(fill=False),corner=True)


# In[20]:


data.groupby(['Outcome']).mean().T


# In[21]:


y=data['Outcome']
x=data.drop(['Outcome'],axis=1)


# In[23]:


test_data=data.tail(100)


# In[24]:


y=data['Outcome']
x=data.drop(['Outcome'],axis=1)


# In[25]:


y


# In[26]:


x


# In[27]:


x.shape


# In[28]:


y.shape


# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2,random_state=0)


# In[30]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[31]:


from sklearn.linear_model import LogisticRegression
LG=LogisticRegression()
LG.fit(x_train,y_train)
Y_LG=LG.predict(x_test)


# In[32]:


Y_LG


# In[33]:


from sklearn.metrics import accuracy_score


# In[34]:


A=accuracy_score(Y_LG,y_test)


# In[35]:


A*100


# In[36]:


from sklearn.metrics import precision_score,recall_score,f1_score,recall_score
def metrics(actuals,predictions):
    print("Accuracy:{:.5f}".format(accuracy_score(actuals,predictions)))
    print("Accuracy:{:.5f}".format(precision_score(actuals,predictions)))
    print("Accuracy:{:.5f}".format(f1_score(actuals,predictions)))
    print("Accuracy:{:.5f}".format(recall_score(actuals,predictions)))


# In[37]:


metrics(y_test,Y_LG.round())


# In[38]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,Y_LG)


# In[39]:


from sklearn.tree import DecisionTreeClassifier


# In[40]:


deci=DecisionTreeClassifier()


# In[41]:


deci.fit(x_train,y_train)
prediction_rf=deci.predict(x_test)


# In[42]:


navaaaaaa=deci.score(x_test,y_test)*100
print(navaaaaaa)


# In[43]:


from sklearn.metrics import precision_score,recall_score,f1_score,recall_score
def metrics(actuals,predictions):
    print("Accuracy:{:.5f}".format(accuracy_score(actuals,predictions)))
    print("Accuracy:{:.5f}".format(precision_score(actuals,predictions)))
    print("Accuracy:{:.5f}".format(f1_score(actuals,predictions)))
    print("Accuracy:{:.5f}".format(recall_score(actuals,predictions)))


# In[44]:


metrics(y_test,prediction_rf)


# In[45]:


from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(n_estimators=100)


# In[46]:


random_forest.fit(x_train,y_train)


# In[47]:


prediction_rf=random_forest.predict(x_test)


# In[48]:


random_forest_score=random_forest.score(x_test,y_test)*100


# In[49]:


print(random_forest_score)


# In[ ]:




