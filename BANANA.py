#!/usr/bin/env python
# coding: utf-8

# #                            Banana Quality

# In[76]:


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Encoding
import sklearn.preprocessing 
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
#Traing and testing
import sklearn.linear_model
from sklearn.model_selection import train_test_split
#Development
from sklearn.linear_model import LinearRegression
linear_regression_model =LinearRegression()
#Evaluation
import sklearn.metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,r2_score,f1_score,recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[38]:


data =pd.read_csv("C:/Users/Oooba/Desktop/Analysis with pyhton/banana_quality.csv")
data


# In[39]:


data.info()


# In[40]:


data.isnull().sum()


# In[41]:


data.describe()


# In[42]:


data_Quality=data["Quality"].value_counts()
data_Quality


# In[67]:


plt.pie(data_Quality ,labels=['Good', 'Bad'], autopct='%1.1f%%', explode=[0,0.1],shadow=True)
plt.title('Good & Bad')
plt.legend()
plt.show()


# # Dependent and Independent

# In[44]:


data["Quality_lab"]=le.fit_transform(data["Quality"])


# In[45]:


data_corr= data.corr()
sns.heatmap(data_corr,annot=True,fmt="0.1f",linewidth=0.5)


# In[65]:


sns.countplot(x="Quality_lab",data=data,palette=["r","g"],alpha=0.6)


# #  Splitting and Traning Testing

# In[82]:


x= data[["Size","Weight","Sweetness","Softness","HarvestTime","Ripeness","Acidity"]]
y= data["Quality_lab"]


# In[47]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape #80%


# # Model Development and predection and Error

# In[70]:


model=linear_regression_model.fit(x_train,y_train) 
model


# In[78]:


y_prede=model.predict(x_test)
y_error= y_test-y_prede
predection=pd.DataFrame({"Actual":y_test,"predicted":y_prede,"Error":y_error})
predection["abs_error"]=abs(predection["Error"])
mean_absolut_error=predection["abs_error"].mean()
predection.head(10) 


# # Model Accuracy and Evaluation

# In[50]:


r2_score(y_test,y_prede)
print(f"Accuracy of the model={round(r2_score(y_test,y_prede)*100)}%")


# In[58]:


print("Root Mean Squared Error (RMSE)=",mean_absolut_error**(0.5))


# # cofficients

# In[95]:


model_cof=model.coef_
plt.plot(model_cof,color="g",marker="*",markersize=12)
plt.title("Cofficient of Model")


# In[96]:


I=model.intercept_
print(f"intercept of the model={round(I*100)}%")


# In[97]:




