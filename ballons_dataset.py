import numpy as np
import pandas as pd

dataset=pd.read_csv('balloons_dataset.csv')

from sklearn.preprocessing import LabelEncoder
lbl_color=LabelEncoder()
dataset['color']=lbl_color.fit_transform(dataset['color'])
lbl_size=LabelEncoder()
dataset['size']=lbl_size.fit_transform(dataset['size'])
lbl_act=LabelEncoder()
dataset['act']=lbl_act.fit_transform(dataset['act'])
lbl_age=LabelEncoder()
dataset['age']=lbl_age.fit_transform(dataset['age'])
lbl_inflated=LabelEncoder()
dataset['inflated']=lbl_inflated.fit_transform(dataset['inflated'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))