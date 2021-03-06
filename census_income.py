import numpy as np
import pandas as pd

dataset=pd.read_csv('census_income_dataset.csv')

from sklearn.preprocessing import LabelEncoder
lbl_workclass=LabelEncoder()
dataset['workclass']=lbl_workclass.fit_transform(dataset['workclass'])
lbl_education=LabelEncoder()
dataset['education']=lbl_education.fit_transform(dataset['education'])
lbl_maritalstatus=LabelEncoder()
dataset['marital_status']=lbl_maritalstatus.fit_transform(dataset['marital_status'])
lbl_occupation=LabelEncoder()
dataset['occupation']=lbl_occupation.fit_transform(dataset['occupation'])
lbl_relationship=LabelEncoder()
dataset['relationship']=lbl_relationship.fit_transform(dataset['relationship'])
lbl_race=LabelEncoder()
dataset['race']=lbl_race.fit_transform(dataset['race'])
lbl_sex=LabelEncoder()
dataset['sex']=lbl_sex.fit_transform(dataset['sex'])
lbl_native=LabelEncoder()
dataset['native_country']=lbl_native.fit_transform(dataset['native_country'])
lbl_incomelevel=LabelEncoder()
dataset['income_level']=lbl_incomelevel.fit_transform(dataset['income_level'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

from sklearn.linear_model import LogisticRegression 
reg=LogisticRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)
print(reg.score(x_test,y_test))