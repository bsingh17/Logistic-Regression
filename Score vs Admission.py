import numpy as np
import pandas as pd

dataset=pd.read_csv('Score vs Admission.csv')
x=dataset.drop('Serial No.',axis='columns')
x=x.drop('Chance of Admit ',axis='columns')
y=dataset['Chance of Admit ']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

print(reg.score(x,y))
