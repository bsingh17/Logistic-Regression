import numpy as np
import pandas as pd

dataset=pd.read_csv('pass or fail.csv')
x=dataset.drop('Pass_Or_Fail',axis='columns')
y=dataset.Pass_Or_Fail

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)

from sklearn import metrics
print(metrics.accuracy_score(y_test,y_predict))