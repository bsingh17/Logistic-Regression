import numpy as np
import pandas as pd

train=pd.read_csv('deodrant liking_train.csv')
test=pd.read_csv('deodrant liking_test.csv')
input_train=train.drop('Instant.Liking',axis='columns')
target_train=train['Instant.Liking']
input_test=pd.DataFrame(test)

from sklearn.preprocessing import LabelEncoder
lbl_trainProduct=LabelEncoder()
lbl_testProduct=LabelEncoder()

input_train['Product']=lbl_trainProduct.fit_transform(input_train['Product'])
input_test['Product']=lbl_testProduct.fit_transform(input_test['Product'])

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(input_train,target_train)

target_predict=reg.predict(input_test)

