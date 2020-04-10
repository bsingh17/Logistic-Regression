import numpy as np
import pandas as pd

dataset=pd.read_csv('bank_marketing_dataset.csv')
dataset=dataset.drop(['marital','education','default','day','month','poutcome'],axis='columns')

from sklearn.preprocessing import LabelEncoder
lbl_job=LabelEncoder()
dataset['job']=lbl_job.fit_transform(dataset['job'])
lbl_housing=LabelEncoder()
dataset['housing']=lbl_housing.fit_transform(dataset['housing'])
lbl_loan=LabelEncoder()
dataset['loan']=lbl_loan.fit_transform(dataset['loan'])
lbl_contact=LabelEncoder()
dataset['contact']=lbl_contact.fit_transform(dataset['contact'])
lbl_y=LabelEncoder()
dataset['y']=lbl_y.fit_transform(dataset['y'])
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=100)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))